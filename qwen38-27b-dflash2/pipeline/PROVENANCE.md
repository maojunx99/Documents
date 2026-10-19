# 源码与公开整理说明

## 已有实现

- 上游项目：[vllm-project/speculators](https://github.com/vllm-project/speculators)。本目录包括本地适配，不能只用上游commit或pip版本替代这些源码。
- `gpu/source/`：从GPU当前实际环境只读获取，Git HEAD为`f0076d415df21fe7ebe3560f90ccad0b063ac429`，包括本地未提交改动。
- `regeneration/source/`：从NPU实际`regenerate-responses`命令解析到的环境获取，同一Git HEAD `f0076d415df21fe7ebe3560f90ccad0b063ac429`，单独保留该环境的本地实现。
- `npu/source/`：来自已完成的FSDP4/512 anchors full_01归档，基于`7a58fc56217632d8d179b665734fa2269e8d9ffa`，包含NPU算子、SDPA、训练/检查点适配。其CLI与重采样环境不同。
- `gpu/continuation/`：文章最终模型的实际补充训练代码及传递依赖，含原生模型的目标与边界检查补丁。

hs_connectors源码与包元数据随相应快照保留；各快照中的`LICENSE`为上游Apache-2.0许可证。保留原文件内的版权与出处。

`CAPTURE_MANIFEST.json`记录本次获取的原文件hash；`SOURCE_MANIFEST.json`记录最终公开文件hash。前者对应获取时的字节，后者包含公开整理修改，不应混为一份完全未改动的运行快照。

## 公开整理修改（2026-10-19）

1. 只收集源码、配置、环境包版本和协议；排除模型、数据、回答、hidden缓存、运行日志、SSH/远程登录脚本及历史任务身份。
2. 机器用户名与私有工作路径改为`/workspace`、`/models`占位；需要改动的源文件带有Publication change注释。上游版本记录保留。
3. 新增`common/managed.py`、GPU基础训练和HTTP重采样的公开入口。其命令取自已有脚本，并增加新输出目录限制、命令/身份记录及owned清理；这些是新的编排层。
4. GPU教师路由算法保留，端口与日志位置改为环境变量。旧GPU smoke脚本的删除输出行为改为拒绝覆盖，`gpu/reference/`仅供溯源。
5. NPU FSDP训练主体保留；补齐原先来自旧目录的选数、CPU检查点和预检脚本，Python/模型/数据路径改为环境变量，并使SIGTERM进入原有finally清理。公开入口重新生成预检收据，未复制历史passed状态。
6. NPU完成审计支持首次prepare的文件指纹、当前run的目标文件清单及配置中的设备号，不依赖旧run_03或固定历史prepared路径。
7. 最终模型后续实验的固定协议仅用于解释算法与预算；迁移时仍需真实数据和起点，并重新封存hash，不能绕过断言直接开跑。

NPU历史完整验证：10000条、FSDP4、512 anchors、packed4096、各rank909步，训练/原生验证/模型及优化器真实回载/owned退出通过。它不证明NPU和GPU数值完全等价，也不证明公开编排层已经再次通过完整硬件运行。

上述原始公开整理只做代码获取、整理和CPU静态检查。后续用户要求增加可配置的x×FSDPy和一键流程，另增加了`examples/train/npu_dflash2_27b.sh`、`npu/recipe.py`、`npu/run_pipeline.py`、通用并行预检/训练审计及独立完成检查；没有修改原训练实现快照。

通用并行包装在运行时设置replicate×shard的DeviceMesh、非重入activation checkpointing，以及记录真实更新/有限梯度/复制组参数与AdamW状态。损失、采样器、每轮训练和验证仍调用native实现；有界`max_steps`的外层循环在累计上限后退出，不会意外继续下一轮。检查点仍使用原生BF16格式，实际回载不代表FP32无损恢复。

重生成入口新增可配置采样参数与目标文件身份，按conversation primary ID校验完整覆盖，允许多轮/工具对话产生多条监督记录。进程清理支持native launcher执行`exec`后argv变化，但仍核对内核starttime、私有session和清理时的当前argv，并通过实际Linux进程测试验证不影响无关进程。新的硬件测试均使用独立新运行目录，不覆盖既有结果、模型或原数据；具体预算与结果以`VALIDATION.json`为准。

## 分平台文档与NPU分步入口（2026-10-19）

运行说明拆为`gpu/README.md`和`npu/README.md`，主README只保留平台导航，旧1万条入口移到`npu/LEGACY_FSDP4.md`。新增`examples/npu/01_container.sh`至`04_run.sh`，分别负责容器、依赖、独立配置及运行；训练步骤仍委托已有的统一管线，不改损失、训练循环或原生源码快照。配置模板默认采用8卡主机上的FSDP4＋TP2教师布局，x/y与卡号可编辑。

安装步骤锁定镜像计算栈、补充训练依赖并检查源码加载；正式与smoke调用读取同一配置，小数据验证只在当前子进程覆盖预算。新增脚本通过Bash语法、无设备命令规划、配置/输出防覆盖、采样/拓扑传递及输入保护检查。没有新增硬件训练验证；官方干净镜像的端到端部署仍待在目标机器上验证。

## 独立训练阶段与真实推理评测（2026-10-19）

保留默认一键训练逻辑，另增加`examples/npu/stages/`五个入口和`npu/stages.py`，在固定配置/运行目录中独立执行预检、重生成、预处理、教师/训练和评测。阶段交接核验完成记录、源代码/数据产物及目标身份；已完成阶段不会重复派发，失败现场不覆盖。同一任务的并发派发由独占锁阻止。

`evaluation/`是新公开编排层，读取用户自备的五基准case清单，按T0/T1、C1、BF16/spec7/Eager、thinking+xhigh进行真实HTTP推理；输入只含原始user消息。保留每条请求的request/response/usage/finish reason与前后Prometheus metrics，并在汇总前重新复算接受长度恒等式。原生验证和检查点实际回载仍由训练阶段执行，不能用它们代替真实推理指标。

serving导出只生成独立配置与原权重链接，不改原检查点。配置转换已用实际NPU训练检查点与H200已安装vLLM的`update_dflash2`逐项CPU核对一致，包括hidden-state/decoder-layer索引转换与非因果SWA。实际253训练环境的vLLM0.26未注册DFlash2；新的评测入口允许另设`EVAL_PYTHON`，不自动修改或升级运行环境。

新增阶段交接、原始case兼容性、参数/防覆盖及模拟HTTP/计数审计均已检查。未用新增分阶段入口重跑NPU训练或真实五基准；CPU测试的模拟请求不能当作硬件推理结果。
