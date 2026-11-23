# NPU：从零训练与已有权重微调

默认随机初始化DFlash2草稿，目标模型冻结。以27B同结构模型、910B/A2为蓝本；Qwen3.6-27B等可更换目标权重后重新生成数据。

在`pipeline/`目录按下表执行。每一步独立脚本，完整训练也保留一键入口。

| 步骤 | 执行位置 | 调用 |
|---|---|---|
| 容器 | 主机 | `bash examples/npu/01_container.sh` |
| 安装依赖 | 容器 | `bash examples/npu/02_install.sh` |
| 生成配置 | 容器 | `bash examples/npu/03_configure.sh` |
| 编辑参数 | 容器 | 修改`config.local.npu.sh` |
| 查看计划 | 容器 | `bash examples/npu/04_run.sh --dry-run` |
| 小步验证 | 容器 | `bash examples/npu/04_run.sh --smoke` |
| 完整运行 | 容器 | `bash examples/npu/04_run.sh` |

已有容器可直接复用；不必重新拉镜像或创建同类容器。安装脚本加载仓库内Speculators/hs-connectors适配源码并锁定计算栈，不要另行升级torch/vLLM。默认A2镜像`quay.io/ascend/vllm-ascend:v0.26.0rc2`；完整已验证版本及干净镜像的验证边界见[环境与审计细节](DETAILS.md#7-验证范围与参考)。

<a id="from-scratch"></a>

## 从零训练：只改一份配置

| 配置项 | 含义 |
|---|---|
| `TARGET_MODEL`、`INPUT_FILES` | 目标权重、原始题源JSONL；重生成不额外加入system prompt |
| `FROM_PRETRAINED=""` | 留空即从随机草稿开始；无需官方draft权重 |
| `FSDP_REPLICAS`、`FSDP_SHARD_SIZE`、`TRAIN_DEVICES` | x×FSDPy及训练卡；教师另用空卡 |
| `REGEN_TP_GROUPS` | 重生成的独立TP组；生成退出后可复用这些卡 |
| `TEACHER_TP_GROUPS` | 仅本机教师，例如`4,5;6,7`表示两组TP2 |
| `TEMPERATURE`、`TOP_P`、`TOP_K`、`PRESENCE_PENALTY`、`REASONING_EFFORT` | 回答重采样参数 |
| `SAMPLE_MAX_LENGTH`、`PACKED_TOKENS`、`MAX_ANCHORS` | 单条上限、packed预算、每rank批次共享anchor预算 |
| `EPOCHS`、`LR`、`WARMUP_RATIO` | 学习率计划；默认5轮、6e-4、4% |
| `SELECTOR_LOSS_TYPE`、`SELECTOR_ALPHA` | selector 目标及权重；默认 `hard_ce`、0.1，可选 `full_teacher_tv` |
| `REGENERATED_DIR` | 可选复用同一目标身份核验通过的已有回答 |
| `RUN_DIR` | 新目录；留空自动创建，拒绝覆盖旧任务 |

默认FSDP4用0～3卡、教师TP2用4/5卡，单条/packed4096、512 anchors、CE.1+TV.9、selector alpha.1。只执行epoch0但保留五轮调度时设置`EPOCHS=5`、`STOP_AFTER_EPOCH=0`。

### 可选：匹配教师概率分布

在同一份 `config.local.npu.sh` 中设置 `SELECTOR_LOSS_TYPE=full_teacher_tv`、`SELECTOR_ALPHA=0.1`，一键入口和分阶段训练都会传递这个选择。Python/原生训练入口使用 `--selector-loss-type full_teacher_tv`；从零训练和 `FROM_PRETRAINED` 微调均适用。默认仍为 `hard_ce`。

该目标最小化 `1 − Σ min(p_teacher, q_draft)`：教师采用完整词表的 T=1 概率，草稿概率仅在原有 top16 内归一化，教师不在 top16 内重新归一化。只替换 selector 损失，保留原有候选/缺失 argmax 注入、unary CE.1＋TV.9、mask、位置衰减及所有草稿参数的梯度。全词表归一化分块计算，避免整份 logits 同时转成 FP32。原生 CLI 若使用 D-PACE，仍由 hard-target CE 置信度计算位置权重。

这是可选实验目标，不能保证真实推理接受长度提升。新增分支已完成 CPU 数值、梯度、原生配置和命令传递验证，尚未做新增 NPU/GPU 端到端训练验证。

## 多机器教师：仍只改这份配置

`TEACHER_NODES`非空时替代`TEACHER_TP_GROUPS`，把本机教师也写进去。每行是**机器或SSH别名｜物理卡号｜TP｜现有容器名（可选）**：

```bash
TEACHER_NODES=(
  "local|4,5|2"
  "10.0.0.12|8,9,10,11|2|existing-teacher"
)
TEACHER_SHARED_ROOT=/mnt/shared/dflash2
TEACHER_SSH_USER=your_user
```

以上是本机1组TP2＋远端2组TP2。远端默认容器、Python、模型路径可用`TEACHER_CONTAINER`、`TEACHER_PYTHON`、`TEACHER_MODEL`统一配置；每行第四项可覆盖容器名。**复用现有环境，不自动安装依赖、建容器或配置SSH凭据**。训练容器里的SSH须已能非交互连接所列主机，主机用户须能`docker exec`现有容器。

所有容器必须挂载同一个可读写共享目录、相同绝对路径，并具备兼容的依赖。脚本自动快照小体积代码、拆分TP组并创建新任务目录，训练阶段直接启动原生teacher；正常结束只释放本次进程。**没有认证令牌、独立网关或create_key步骤。**

共享盘通过原生`shared_file`后端传hidden-state文件：先保存临时文件，fsync后原子重命名，trainer只读取最终文件，不使用跨机flock。HTTP传请求和文件句柄；SSH只管远端启动/退出，不传tensor。如果教师服务由平台预先启动，可用[直接HTTP接入](../teachers/README.md#预启动原生服务)，不需要trainer再SSH。详细协议与限制见[教师说明](../teachers/README.md)。

<a id="from-pretrained"></a>

## 已有权重微调

在同一配置中设置`FROM_PRETRAINED=/models/your_dflash2`，其余调用不变。支持官方发布格式与原生checkpoint；原目录只读，自动适配配置并核验全部初始张量。`LR`仍使用配置中的值，请为本次微调明确设置。

该入口新建AdamW与scheduler；不是恢复源optimizer/RNG的断点续跑。格式转换、FP32主参数与BF16计算说明见[完整细节](DETAILS.md#from-pretrained)。

<a id="staged-run"></a>

## 分阶段运行与结果

配置中填写固定的新`RUN_DIR`，依次调用：

```bash
bash examples/npu/stages/01_preflight.sh
bash examples/npu/stages/02_regenerate.sh
bash examples/npu/stages/03_prepare.sh
bash examples/npu/stages/04_train.sh
bash examples/npu/stages/05_evaluate.sh
```

前四步包括原生训练验证/保存/回载；第五步额外运行T=0和T=1真实推理五基准，需配置`EVAL_CASES`及支持DFlash2的推理环境。[各阶段说明](STAGES.md)

成功须`COMPLETION_VERIFICATION.json: passed=true`且owned清理通过。失败查`failure.json`、阶段日志和`audit/failure_rank*.json`，保留现场，不覆盖重跑。[训练审计与验证边界](DETAILS.md) · [环境记录](environment-recorded.json) · [平台选择](../README.md)
