# NPU分阶段运行

先按[NPU环境说明](README.md)完成容器、依赖和配置。所有命令从`pipeline/`目录执行；与一键入口使用同一份`config.local.npu.sh`。

将`RUN_DIR`填为固定的新目录，五次调用都使用这个目录；运行中不要改变模型、题源、拓扑或训练参数。每步都可加`--config /path/to/config.sh`或`--dry-run`。

| 阶段 | 调用 | 完成后保留的产物 |
|---|---|---|
| 预检 | `bash examples/npu/stages/01_preflight.sh` | 模型/tokenizer兼容性、NPU算子、checkpointing及x×FSDPy通信/梯度检查 |
| 重生成 | `bash examples/npu/stages/02_regenerate.sh` | 目标回答、采样配置、完整性校验；生成服务已停止 |
| 预处理 | `bash examples/npu/stages/03_prepare.sh` | prepared数据、长度/监督量审计、文件指纹 |
| 教师/训练 | `bash examples/npu/stages/04_train.sh` | 每轮原生验证、检查点、实际回载检查；教师与训练进程已清理 |
| 真实推理评测 | `bash examples/npu/stages/05_evaluate.sh` | T=0/T=1五基准逐请求原始结果、接受长度与性能汇总 |

## 阶段交接

前四步的完成记录在`RUN_DIR/stages/<阶段>/COMPLETED.json`。下一步核验配置、前一步产物、目标模型身份及源码指纹，不会静默重做前一步。重复调用已完成阶段只做核验；失败或未完成目录保留，禁止覆盖或自动重新开训。`.stage.lock`阻止同一任务并发派发，异常遗留锁须先核实进程再处理。

`REGENERATED_DIR`可指定已经通过同一target身份核验的回答，第2步仅核验复用，随后仍需第3步。训练阶段自动启动并停止独立在线教师，原生验证仍随每轮训练执行，不必单独保留一个长期运行的教师服务。

一键训练完成后也可调用第5步：把`RUN_DIR`指向已经通过`COMPLETION_VERIFICATION.json`的任务目录。它不会重新训练。

## 五基准评测配置

| 参数 | 默认值 / 含义 |
|---|---|
| `EVAL_CASES` | 自备五基准JSON/JSONL题目清单，格式见[评测说明](../evaluation/README.md) |
| `EVAL_CHECKPOINT` | 留空使用最后一个完成的epoch；也可明确指定检查点 |
| `EVAL_REFERENCE` | 可选官方/社区DFlash2权重，与本轮权重使用同一批提示和seed比较 |
| `EVAL_PYTHON` | 支持DFlash2的NPU推理环境，可与训练Python不同 |
| `EVAL_TP_GROUPS` | `0,1;2,3;4,5;6,7`，四个TP2服务并行分题，每实例C1 |
| `EVAL_PER_BENCHMARK` | 每基准32条；单权重共160条×2种温度=320请求 |
| `EVAL_MAX_TOKENS`、`EVAL_CONTEXT` | 4096 / 8192，不静默裁切提示 |
| `EVAL_PRESENCE_PENALTY` | 0，与重生成阶段的presence1.5分开配置 |
| `EVAL_OUTPUT` | 留空每次新建`RUN_DIR/evaluations/`下的结果目录，可显式指定新目录 |

第5步总是分别运行T0（greedy draft）和T1（top-p.95/top-k20、probabilistic draft），BF16/spec7/Eager、thinking+xhigh，不手工添加system prompt。它可以独立重复调用，每次保留新的结果和服务身份；不会覆盖前次评测，也不会以训练acc/loss代替真实请求。

成功记录为本次评测目录的`COMPLETED.json`，汇总在`summary.json`；所有request/response、usage、finish reason、前后Prometheus metrics和逐请求结果都保留。可选参考模型时还输出按提示配对的接受长度差和bootstrap区间。这是接受长度与推理性能测试，不包括答案正确率或代码pass@1。

## 推理环境与验证边界

已只读核实253当前训练环境的vLLM0.26仅注册DFlash1、未注册DFlash2；不能直接用它评测DFlash2。请把`EVAL_PYTHON`指向已包含DFlash2模型、selector与概率草稿路径的NPU环境。评测先检查这些能力，缺失即保留失败日志并退出，不回退到DFlash1。

本次新增分阶段入口完成了CPU阶段交接、参数/指纹/防覆盖检查和模拟HTTP指标审计；配置导出还与H200已安装vLLM的DFlash2转换器逐项核对一致。**尚未用这些新增入口重跑NPU训练或真实五基准**，不能把这些检查当作新的硬件通过记录。
