# 真实推理五基准

入口为[NPU第5阶段](../npu/STAGES.md)，使用gsm8k、math500、humaneval、mbpp、mt-bench各32条原始提示，分别运行T0与T1。题目数据不随代码发布；可直接使用已有的五基准case清单。

## 输入清单

`EVAL_CASES`支持JSON列表、包含`cases`列表的JSON对象，或JSONL。每条记录使用以下字段：

| 字段 | 含义 |
|---|---|
| `benchmark` | 上述五个基准名称之一，大小写不敏感 |
| `messages`或`prompt` | 仅一个原始user消息，或原始提示字符串；拒绝额外system/few-shot消息 |
| `case_id`或`id` | 可选；同一基准内不可重复 |
| `seed` | 可选；缺省按清单位置固定生成，所有权重与温度使用同一seed |
| `prompt_hash` | 可选；提供时核验原始用户文本的SHA256 |

每个基准固定取清单中的前`EVAL_PER_BENCHMARK`条，不做基于结果的筛选。数量不足或提示哈希不符就报错，实际选择的题目另存到结果目录。不同温度、模型和TP分片使用同一份选择结果。

## 输出和指标

- `protocol.json`、`runtime.json`、`model_fingerprints.json`：参数、实际推理环境与模型身份。
- `exports/`：独立serving配置及指向原权重的链接，不改原检查点；转换保留特征层索引和非因果注意力语义。
- `worker*/<模型>/t*/cases/`：完整请求、响应、usage、finish reason与前后metrics。
- `summary.json`：每个温度/模型/基准的逐请求平均接受长度、draft加权接受长度、条件接受率、速度及截断数。
- `COMPLETED.json`：真实请求完整覆盖、原始metrics复算、源码/权重未变与进程清理通过。

逐请求接受长度为`1 + accepted_tokens / draft_rounds`；主指标是它的请求算术平均。全局draft加权值另列，不能混用。没有draft round的请求仍保留，tau记为null，汇总明确列出可计算tau的请求数。

位置累计计数必须递减，条件接受率则按相邻位置计数相除，本身不要求单调。每条请求均核验接受token数等于位置计数之和，并与tau恒等式交叉检查。

速度同时给出`总输出token / 请求耗时之和`及逐请求token/s的平均值；计时包含HTTP生成请求，排除服务启动和metrics轮询。多个TP服务并行分题，每实例C1，这些数值不冒充单实例C4或整机最大吞吐。达到输出上限的样本保留并计入截断统计。

T0使用temperature0、top-p1、top-k-1和greedy draft；T1使用temperature1、top-p.95、top-k20和probabilistic draft。两者默认presence0、max_tokens4096、thinking+xhigh。原生chat template可能渲染模型自身的reasoning说明；客户端不额外拼system prompt。

配置和CPU协议检查不代表硬件数值等价；正式结果必须由目标NPU环境完成请求与审计后产生。当前训练环境若只支持DFlash1，必须另配DFlash2推理环境，脚本会在启动前拒绝错误路径。
