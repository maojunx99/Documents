# NPU：从容器到训练

以下以910B/A2、单机8卡为例，命令从`pipeline/`目录执行。主机先准备驱动/固件、Docker、完整目标模型和原始问题JSONL。

| 步骤 | 脚本 | 执行位置 |
|---|---|---|
| 1. 启动容器 | `examples/npu/01_container.sh` | 主机 |
| 2. 安装并检查依赖 | `examples/npu/02_install.sh` | 容器 |
| 3. 生成可编辑配置 | `examples/npu/03_configure.sh` | 容器 |
| 4. 验证或正式运行 | `examples/npu/04_run.sh` | 容器 |

## 1. 启动容器

获取代码，修改`01_container.sh`顶部的主机目录、镜像和卡号，然后运行。容器内自动挂载代码到`/workspace/pipeline`、模型到`/models`、题源到`/data/prompts`、结果到`/data/runs`。

```bash
git clone https://github.com/maojunx99/Documents.git
cd Documents/qwen38-27b-dflash2/pipeline
bash examples/npu/01_container.sh
```

默认镜像是A2的`quay.io/ascend/vllm-ascend:v0.26.0rc2`；A3需选择对应`-a3`镜像。卡号必须空闲，16卡机器需同时修改容器暴露卡号与第3步的训练/教师分组。已有容器用`docker start -ai dflash2-27b-train`进入。只查看启动命令可加`--dry-run`。

## 2. 安装依赖

进入容器后，工作目录已是`/workspace/pipeline`：

```bash
bash examples/npu/02_install.sh
```

脚本加载CANN/ATB环境，创建`/workspace/venvs/dflash2`，锁定镜像的计算栈后安装`requirements-training.txt`，再检查NPU、两个源码快照及仓库完整性。环境记录保存到`/workspace/environment_records`。自定义位置可修改脚本顶部的`VENV_PATH`和`ENV_RECORDS`。

Speculators和hs-connectors直接加载仓库适配源码，无需再从PyPI安装。不要用归档中的`egg-info/requires.txt`安装依赖；基础包冲突时应核对镜像，不能通过升级torch/vLLM绕过约束。

## 3. 修改参数

```bash
bash examples/npu/03_configure.sh
vi config.local.npu.sh
```

配置从[config.example.sh](../examples/npu/config.example.sh)复制；已有配置不会覆盖，`config.local.*`不上传GitHub。后续所有运行读取这个文件，不需要修改训练实现。

| 参数 | 修改内容 |
|---|---|
| `TARGET_MODEL`、`INPUT_FILES` | 容器内模型路径、原始题源文件列表 |
| `FSDP_REPLICAS`、`FSDP_SHARD_SIZE` | x、y；训练卡数=x×y |
| `TRAIN_DEVICES`、`TEACHER_TP_GROUPS`、`REGEN_TP_GROUPS` | 训练卡、独立在线教师、回答生成的TP分组 |
| `TEMPERATURE`、`TOP_P`、`TOP_K`、`PRESENCE_PENALTY`、`REASONING_EFFORT` | 回答重生成参数 |
| `SAMPLE_MAX_LENGTH`、`PACKED_TOKENS`、`MAX_ANCHORS` | 单样本上限、packed预算、每rank packed批次共享的anchors上限 |
| `EPOCHS`、`LR`、`WARMUP_RATIO`、`SELECTOR_ALPHA` | 训练轮数、学习率、warmup和selector损失权重 |
| `RUN_ROOT`、`RUN_DIR`、`STAGE_TIMEOUT` | 结果位置、新任务目录、每阶段时间上限 |

默认FSDP4使用0～3卡、在线教师TP2使用4/5卡；重生成阶段先用4个TP2覆盖0～7卡，退出后再进入训练。2×FSDP4需要8张训练卡，**教师另需独立卡**；所有分组均可配置，改变卡数不会自动调整学习率。

题源每行需含`messages`或`conversations`；已有assistant回答会重新生成。默认T1/p.95/k20/presence1.5/xhigh、输出上限8192，客户端不额外添加system prompt。训练默认单样本/packed4096、512 anchors、5轮、CE.1＋TV.9、selector alpha.1、AdamW6e-4/WD0。

同结构Qwen3.6-27B等权重可修改`TARGET_MODEL`，随后重新生成回答与hidden states；入口检查模型/tokenizer兼容性。不同结构需明确修改draft、特征层及mask配置后重新验证。只有同一目标身份核验通过的已有生成结果，才可用`REGENERATED_DIR`跳过生成。

## 4. 运行

### 一键运行

先查看配置与命令，再用128条原始对话、最多2次更新验证全流程：

```bash
bash examples/npu/04_run.sh --dry-run
bash examples/npu/04_run.sh --smoke
```

smoke只覆盖本次调用的输出长度、划分、warmup和步数，不改配置文件；通过后正式运行：

```bash
bash examples/npu/04_run.sh
```

可用`--config /path/to/config.sh`指定另一份配置，`--smoke --dry-run`仅展示小规模计划。默认自动创建带时间戳的新任务目录；固定`RUN_DIR`时须保证它不存在。完整流程会依次执行预检、重生成、停止生成服务、预处理、教师/训练、每轮验证/保存、实际回载和清理。需要后台运行时用tmux或现有作业系统承载。

成功标志是`status.json`的`stage=completed`，且`COMPLETION_VERIFICATION.json`中`passed=true`。失败查看`failure.json`、`stage_processes/*.log`、`audit/failure_rank*.json`和`cleanup_summary.json`。不要覆盖失败目录或自动重启训练。结果保留到主机挂载目录；五轮检查点需另留约每轮12GiB及数据/缓存空间。

### 分阶段运行

需要分别执行预检、重生成、预处理、教师/训练和真实推理评测时，使用[分阶段说明](STAGES.md)。这些脚本共用`config.local.npu.sh`；一键入口仍保留。

```bash
bash examples/npu/stages/01_preflight.sh
bash examples/npu/stages/02_regenerate.sh
bash examples/npu/stages/03_prepare.sh
bash examples/npu/stages/04_train.sh
bash examples/npu/stages/05_evaluate.sh
```

先在配置中填写固定的新`RUN_DIR`；评测还需填写`EVAL_CASES`和支持DFlash2的`EVAL_PYTHON`。训练阶段保留每轮原生验证与回载检查；第5步另跑T=0/T=1真实推理五基准。

## 验证范围与参考

已在910B2C既有环境通过128条重生成→2×FSDP4/512 anchors真实更新→原生验证/保存/回载→清理；另有[原1万条FSDP4流程](LEGACY_FSDP4.md)。这验证流程与容量，不是默认五轮的质量评测；BF16落盘与缺少RNG快照也不等于FP32无损恢复。

已验证环境为torch2.10.0、torch_npu2.10.0.post4、vLLM0.26.0、vllm-ascend开发提交`155f68974`、CANN9.1.0；[官方rc2镜像](https://github.com/vllm-project/vllm-ascend/blob/v0.26.0rc2/Dockerfile)与它并非逐字节相同，**干净镜像安装路径尚未单独完成端到端验证**，新机器请先跑smoke。安装脚本的版本检查不替代实际训练验证。

[完整包版本](environment-recorded.json) · [硬件验证记录](../VALIDATION.json) · [源码说明](../PROVENANCE.md) · [返回平台选择](../README.md)
