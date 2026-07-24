## 1. 背景与目标

MTP（Multi-Token Prediction）投机解码的一轮 steady-state decode 包含草稿模型生成、主模型批量验证、拒绝采样，以及下一轮草稿输入准备。优化前这些阶段在 Host 上基本串行：主模型验证完成后，worker 同步设备，把接受结果拷回 CPU，更新缓存和输入，再由 scheduler 发起下一轮 draft forward。设备在整段 Host 工作期间无任务可执行，形成空泡。

本文描述 xLLM 的 MTP 异步流水，包括状态语义、stream/event 依赖、下一轮首个 draft 的提前下发、融合算子、批次切换降级和正确性保障。

设计目标：

- 下一轮首个 draft 在当前轮 `validate` 尾部完成 launch，不等待 scheduler 返回。
- 接受状态尽量驻留设备，避免 D2H → CPU 计算 → H2D 串行链路。
- 只保留真正的跨 stream 事件；同 stream 依赖使用 FIFO 顺序。
- 不改变拒绝采样、KV cache 和随机采样语义，不能以吞吐换接受率。
- 批次切换、首次 decode 和不支持的配置自动回到保守路径。

## 2. 优化前后的流水

### 2.1 优化前：同步边界制造空泡
<img width="1726" height="722" alt="schedule_overlap_base" src="https://github.com/user-attachments/assets/0a43b986-f892-4bd8-bbff-a016d3cd98df" />
问题不只是一次 memcpy。Host 必须等设备结果才能构造输入，一个同步点会把后续 CPU 操作、H2D 和 kernel launch 全部串在设备计算之后，空泡长度等于整段 Host 临界路径。

### 2.2 优化后：当前轮尾部提前下发下一轮
<img width="2786" height="927" alt="mtp_async_optimized" src="https://github.com/user-attachments/assets/bb85b6dc-fb6f-4d01-ad27-e487e7325c20" />
拒绝采样产出的 tensor 不先交给 CPU，而是在 compute stream 上直接准备下一轮输入并 launch draft。scheduler 仍会获得 CPU token，但其工作与设备任务重叠。

## 3. 优化方向一：提前下发下一轮 draft-0

### 问题

优化前，下一轮的第一个 draft forward 必须等 `run_validate()` 返回 scheduler、scheduler 完成批次组织、再把输入下发到设备后才 launch。这段 Host 临界路径（CPU 计算 + H2D + kernel launch）整段挂在设备计算之后，形成空泡。

### 方案

在当前轮 `run_validate()` 返回 scheduler **之前**，就在 compute stream 上提交下一轮的 draft-0 forward。下一轮 `step_decode()` 进入时，draft-0 已经在流上排队甚至执行完，只需直接消费结果，不再重复 launch。

为此引入 `PendingDraftContext`——一个跨 scheduler turn 保存的**预提交 draft-0 结果槽**：

```text
本轮 run_validate() 尾部
    |-- 拒绝采样得到 validate_output
    |-- enqueue_next_first_draft()
    |      |-- KV 补齐 forward（仅写 KV，不采样）
    |      `-- 下一轮 draft-0 forward（正常采样）
    `-- 结果存入 PendingDraftContext

控制权返回 scheduler

下一轮 step_decode()
    |-- 校验 batch identity
    `-- draft_idx == 0 直接消费 PendingDraftContext，不再 launch
```

`PendingDraftContext` 同时持有 `prepared_input`，保证异步执行期间 input tensor 与 graph buffer 的生命周期。batch identity 变化时，worker 等已排队工作结束后丢弃旧结果，防止错误 batch 消费或 buffer 提前释放。

## 4. 优化方向二：接受状态驻留设备，延迟 D2H

### 问题

优化前主模型验证完成后，worker 同步设备、把接受结果拷回 CPU、更新 `EmbeddingCache`、再由 scheduler 构造下一轮输入。这是 `D2H → CPU 计算 → H2D` 的串行链路，挡在设备关键路径上。

### 方案

接受状态**不先交给 CPU**，而是驻留设备，在 compute stream 上直接构造下一轮输入并 launch。scheduler 最终仍需要 CPU token，因此 D2H 不能删除，但采用“提交但不立即等待”：

1. compute stream 把接受 token 异步拷到 pinned Host buffer，紧接着记录 handoff event。
2. 设备 tensor、Host buffer view、event 存入 `PendingTargetContext`——一个 **target 验证结果交接槽**。
3. 立即用设备上的接受状态准备并 launch 下一轮 draft。
4. 只有 CPU 写 `EmbeddingCache` 或 batch transition 时才等待 event。

提前下发路径直接用 `validate_output` 构造设备输入；`PendingTargetContext` 保留在侧，负责稍后把相同结果提交到 Host cache。Host 提交与设备下一轮 draft 完全重叠，不再挡 NPU。

pinned buffer 按容量复用；任意时刻只允许当前 `PendingTargetContext` 持有其 view，复用前必须先 flush 并清空旧槽。

## 5. 优化方向三：固定形状模板与异步准备

### 问题

runner/graph 在 launch 前需要 Host 元数据确定 shape，但此时接受长度对 Host 不可见（结果还在设备上）。无法提前 launch。

### 方案

先在 prepare stream 上构造**保守固定形状模板**，再在 compute stream 上用真实接受状态**覆盖内容张量**：

- 模板用**最大接受偏移**（`num_speculative_tokens`）做 position/KV 长度规划，保证 graph 形状确定。
- token 用安全占位值（负值改 0），embedding 用 placeholder。
- shape 和持久化 buffer 在 prepare stream 上先完成，不依赖接受结果。
- 真实 token、embedding、position、KV、cache slot 随后在 compute stream 上全量覆盖。

模板**只决定执行形状，不能成为数值真相**——所有影响语义的 device tensor 都必须被接受状态替换。这样 prepare 与 target 验证可以并行，模板在 target 跑完时已就绪，compute stream 上立即用真实接受状态 patch 内容即可 launch。

## 6. 优化方向四：融合准备算子

### 问题

从接受状态推导下一轮 draft 输入需要多次比较、归约、gather、where、stack、类型转换和 cache-slot 映射，逐 op 执行 launch 开销显著。

### 方案

`mtp_prepare_next_draft` 把接受状态解析和五组输出合为一个 AscendC launch：

```text
输入: accepted_tokens[B,W] | accepted_embeddings[B,W,H] | embedding_placeholder[H]
      base_positions[B] | base_kv_seq_lens[B] | block_tables[B,N] | block_size

输出: token_ids[B*2] | embeddings[B*2,H] | positions[B*2]
      kv_seq_lens[B] | cache_slots[B*2]
```

输出行顺序为 `[KV 补齐行, 下一轮 draft-0 行]`。当前 kernel 单 AIV core 处理所有 batch 行（输出小，多核标量写可能冲突同一 cache line，launch 延迟是主要开销，单核优先保证确定性）。

不满足融合条件时（device/dtype/维度/对齐/`block_size`/multi block tables/后端未编译）自动降级到 Torch fallback（`mtp_async_state`），调用方只判断 wrapper 是否返回结果，不复制资格条件，避免 worker 与 wrapper 漂移。`mtp_async_state` 同时是融合核的测试参考实现。

## 7. 优化方向五：同步边界最小化与安全降级

### 同 stream 用 FIFO，跨 stream 才用 event

| 依赖类型 | 处理方式 |
| --- | --- |
| 同 stream producer→consumer | FIFO 顺序，不创建 event |
| 跨 stream prepare→compute | record/wait event |
| 设备→Host async D2H | 真正读取时才 wait event |

不使用 `synchronizeStream` 表达普通算子先后关系。draft `i` 的采样输出直接成为 draft `i+1` 的 device tensor，不再加 `EventRecord/EventWait`。同步只在非 steady-state 边界允许：batch identity 改变需丢弃 prelaunch、Host fallback 读 pending D2H、event 创建失败兜底、worker 结束排空。

### Tensor 生命周期

异步 launch 返回不代表设备读完 input。`ForwardOutput` 持有 `retained_input` 和依赖列表：draft input 保留到输出完成；target 继承所有 draft input；提前下发的 draft-0 output 同时继承 KV 补齐 forward 的 input；完成事件被消费后才释放，防止 allocator 复用仍在访问的 tensor，也是 graph replay 地址稳定性的必要条件。

### Batch identity 与降级

提前下发结果只对相同序列集合和顺序有效，以 `embedding_ids + request_ids` 标识 batch：

```text
首次/批次变化 -> Host fallback -> Primed -> async steady state
                       ^                         |
                       └── identity 改变 ───────┘
```

首次 handoff 先通过 Host cache 建立 scheduler 与设备一致的 position/KV base，再把 batch 标记为 primed，之后才完全由设备推导。identity 不匹配时：丢弃旧 `PendingDraftContext`、flush `PendingTargetContext` 到 Host cache、重建 primed identity、用保守路径完成本轮。批次变化是慢路径，同步不落入稳定 decode 关键路径。

## 8. 启用条件与小结

完整提前下发要求：

```text
enable_schedule_overlap
AND target 使用 chunked-prefill speculative verify 路径（Qwen3.5 / MiMo）
AND dp_size <= 1
AND 未启用 ATB speculative kernel
```

不满足只降级，不影响功能。dp 和 ATB speculative kernel 尚未覆盖相同交接约束，保持保守路径。

**小结**：本次优化以“把下一轮 draft-0 提前到当前轮 validate 尾部 launch”为 headline，配套五个方向协同消除设备空泡——接受状态驻留设备、固定形状模板提前准备、KV 补齐 forward 补全因果前缀、融合算子压缩准备开销、同步边界最小化与安全降级。核心原则是：同 stream FIFO 替代 event、D2H 移出关键路径、形状确定性与数值真相分离、batch identity 守护预提交结果正确性。

## 9. 下一步优化方向：草稿模型全图 capture

### 仍存在的空泡

第 3 节的提前下发只消除了**轮间** draft-0 的启动空泡。同一轮内，draft `i` 到 draft `i+1`（`i = 0..N-2`）之间仍有 host 介入空泡：

- `broadcast_spec_tokens`：多 rank 下跨 rank 统一 draft token，是同步集合通信，host 阻塞等待。
- `process_draft_sample_output`：launch 一个 gather kernel 压缩 `[batch, vocab]` probs 进 cache。
- `set_token_ids_device_tensor`：launch `safe_to` 把 int64 token 转 int32 作为下一轮输入。
- `prepare_draft_inputs`：host 构造 position/KV/cache_slots 元数据 + 跨 stream event wait。

根因是 draft `i+1` 的输入 token 来自 draft `i` 的 sampler 输出，算法上不可提前；而当前实现里每次 draft forward 都是独立 eager launch，轮次之间靠 host 串联传递 token 句柄，host 执行期间设备空转。

### 方案：把 draft 循环 capture 进一张图

把 `num_speculative_tokens` 次 draft forward + 它们之间的 token 传递、probs 压缩、broadcast 全部录进一张 graph，replay 时设备内自循环，host 不介入。

参考 vllm-ascend 的 merged-draft 全图模式（`CUDAGraphMode.FULL`）：

1. **预分配静态 buffer**：`input_ids`、`positions`、`hidden_states`、`draft_token_ids_tensor[N]`、`slot_mapping_group[N]`、`seq_lens_group[N]` 等按 `max_num_reqs` 预分配，地址在 graph replay 期间稳定。
2. **capture 阶段预建 per-step attn metadata**：每个 draft step 用独立的 `slot_mapping` / `seq_lens` / `query_start_loc` buffer，capture 时一次性为所有 step build 完 attn metadata，存进 `multi_steps_attn_metadata`。
3. **图内 token 传递**：`draft_token_ids_tensor[draft_index]` → `self.input_ids` → `self.hidden_states` 全是图内静态 buffer 之间的 copy + argmax，无 host 介入。
4. **图内 broadcast**：多 rank 一致性用图内 all_gather 替代 host 侧 `pg->broadcast`。
5. **replay**：一次 graph replay 跑完整个 draft 循环，设备无空泡。

## 10. 设计insights
1. cpu侧同步拷贝尽量消除，无法消除的变同步为异步，做到计算掩盖
2. 异步调度优先，并且数据准备操作从host侧下沉到device侧（device上同一个stream天然保序）
相关资料参考： https://docs.vllm.com.cn/en/latest/design/model_runner_v2/

## 11. 优化通用性与适配
1. 异步调度逻辑本身与模型无关，模型适配点集中在参数准备(todo: 对应代码段)
2. todo: glm5/ds v4适配过程

  

