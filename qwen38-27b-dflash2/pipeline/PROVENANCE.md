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

本次上传只做代码获取、整理和CPU静态检查；没有启动训练、教师或生成服务，没有修改任何原运行目录、模型或数据。
