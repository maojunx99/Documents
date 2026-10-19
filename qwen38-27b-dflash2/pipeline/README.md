# Qwen3.8-27B DFlash2：GPU / NPU完整训练管线

这里保存实际使用过的训练实现、回答重采样实现和公开运行入口。包含原生模型、CE/TV算子、隐藏特征连接器、预处理、教师服务、训练、验证、检查点保存与检查代码。模型权重、训练样本、生成回答和设备驱动需要另外准备。

| 路线 | 已有运行验证 | 数据与主要配置 | 入口 |
|---|---|---|---|
| GPU基础训练 | H200上完成基础训练 | 4张训练卡DDP4＋4个TP1教师；单样本4096、packed8192、1024 anchors；5轮 | `gpu/run_pipeline.py` |
| NPU可配置完整流程 | 支持回答重生成直至训练、验证、检查点与清理；硬件验证范围见下文 | x×FSDPy＋独立TP教师；默认单样本/packed4096、512 anchors | `examples/train/npu_dflash2_27b.sh` |
| NPU原1万条流程 | Ascend 910B2C上完成1万条、1轮；每rank 909步，训练/验证/实际模型及优化器回载通过 | FSDP4＋一个TP2教师；单样本/packed4096、512 anchors | `npu/scripts/run_pipeline_fsdp.py` |
| GPU后续实验 | 产生文章最终评测权重的补充数据训练实现 | 既有模型起点、DDP2、4次梯度累积、FP32 master/AdamW；分布匹配目标 | `gpu/continuation/` |

GPU基础训练与NPU流程验证使用同一类五层草稿结构、完整248320词表和CE/TV目标，但数据预算、packed长度、anchors、并行方式不同。NPU这次验证的是流程与容量，不代表完成了GPU五轮训练的数值复现。

本目录在原归档上补齐依赖，并提供统一NPU入口。原GPU公开编排层仅做过静态与CPU检查；NPU可配置入口的独立验证范围记录在`VALIDATION.json`，不把流程测试当作长程收敛或推理质量验证。原实现与修改范围见[PROVENANCE.md](PROVENANCE.md)。

## 目录

```text
common/managed.py            进程身份、命令记录和仅清理自身子进程
examples/train/
  npu_dflash2_27b.sh         修改顶部变量后，一条命令运行NPU完整流程
regeneration/                GPU/NPU共用的HTTP回答重采样客户端与服务入口
  source/                    NPU实际重采样环境的Speculators源码快照
  reference/lane_worker_pb.sh 原任务队列脚本，供核对历史参数
gpu/
  source/                    GPU训练环境的Speculators及hs_connectors快照
  run_pipeline.py            预处理→4教师→路由→DDP4训练及原生验证→清理
  reference/                 原GPU分阶段脚本，已隐藏机器私有路径
  continuation/              最终模型后续实验代码与协议，不属于从零训练入口
npu/
  run_pipeline.py            重生成→prepare→教师→x×FSDPy→验证/保存/回载→清理
  recipe.py                  模型/拓扑配置检查和完整命令计划
  requirements-training.txt 官方NPU镜像上补充的训练依赖
  source/                    成功FSDP4流程的独立pinned源码及NPU适配
  scripts/                   选1万条、预检、FSDP训练、回载检查与归档
  draft_config.json          显式草稿结构
check_bundle.py              无设备的静态与文件完整性检查
```

## 1. 环境与输入

在Linux机器或已有训练容器内运行。保留机器已有CUDA或CANN环境，尤其是CANN的`PYTHONPATH`；不要让通用依赖安装替换已经配套的`torch`、`torch_npu`、vLLM。

| 环境 | 已记录的关键版本 |
|---|---|
| GPU训练 | PyTorch 2.13.0、vLLM 0.29.0、Transformers 5.16.1、hs-connectors 0.7.1a0 |
| NPU训练 | PyTorch 2.10.0＋torch_npu 2.10.0.post4、Transformers 5.14.1、vLLM 0.26.0、vllm-ascend `155f68974`、CANN 9.1.0 |

完整包版本在`gpu/environment-recorded.json`和`npu/environment-recorded.json`，它们是运行记录，不是可跨平台直接安装的requirements锁文件。Speculators、hs-connectors源码、对应发行元数据及许可证随快照保留，运行入口优先通过`PYTHONPATH`加载；其余第三方依赖从选定的Python环境提供。无需另找旧训练仓库或安装PyPI最新版Speculators。

需要准备：

- 本地完整Qwen3.8-27B目标模型目录，包含配置、tokenizer和全部权重分片。
- 原始问题/对话JSONL，含`conversations`或`messages`。若已有27B重采样结果，直接跳到第3或第4节。
- 输出与hidden-state缓存的可写磁盘。新任务必须使用尚不存在的输出目录。

以下命令均从`pipeline/`目录执行。`PY`为已经配置好对应设备环境的Python。

### 1.1 NPU从新机器开始：容器→依赖→参数→运行

以下以**910B/A2、单机8卡**为例。主机需事先安装匹配的Ascend驱动/固件及Docker，并准备完整目标权重和原始问题JSONL。已在253的既有容器中验证完整管线；下面的**官方干净镜像部署路径尚未单独完成端到端验证**，在新环境上应先跑第4步的小规模检查。镜像版本不能与其它release的torch/CANN随意混装。

#### 第1步：在主机启动官方容器

官方提供A2镜像`quay.io/ascend/vllm-ascend:v0.26.0rc2`；A3需选择对应`-a3`镜像，不能仅根据卡数判断系列。可参照[官方镜像使用说明](https://docs.vllm.ai/projects/ascend/en/v0.26.0rc1/tutorials/models/DeepSeek-V4-Flash.html)及[该版本Dockerfile](https://github.com/vllm-project/vllm-ascend/blob/v0.26.0rc2/Dockerfile)。该镜像包含CANN、PyTorch/TorchNPU、vLLM/vLLM-Ascend和Triton Ascend；主机驱动不由本仓库安装。

在**主机Bash**中修改路径和卡号后执行：

```bash
npu-smi info

export IMAGE=quay.io/ascend/vllm-ascend:v0.26.0rc2
export WORK_ROOT=/data/dflash2_workspace
export MODEL_ROOT=/data/models
export PROMPT_ROOT=/data/dflash2_prompts
export RUN_ROOT=/data/dflash2_runs
export NPU_IDS=0,1,2,3,4,5,6,7
mkdir -p "$WORK_ROOT" "$PROMPT_ROOT" "$RUN_ROOT"
test -d "$MODEL_ROOT" || exit 1  # 此目录应已放好完整模型，例如Qwen3.8-27B/

device_args=()
IFS=',' read -r -a npu_ids <<< "$NPU_IDS"
for card in "${npu_ids[@]}"; do
  test -c "/dev/davinci${card}" || exit 1
  device_args+=(--device "/dev/davinci${card}")
done
mount_args=()
if [[ -f /etc/hccn.conf ]]; then
  mount_args+=(-v /etc/hccn.conf:/etc/hccn.conf:ro)
fi

docker pull "$IMAGE"
docker run -it --name dflash2-27b-train \
  --network host --shm-size=128g --ulimit memlock=-1 \
  "${device_args[@]}" \
  --device /dev/davinci_manager --device /dev/devmm_svm --device /dev/hisi_hdc \
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro \
  -v /usr/local/dcmi:/usr/local/dcmi:ro \
  -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi:ro \
  -v /etc/ascend_install.info:/etc/ascend_install.info:ro \
  "${mount_args[@]}" \
  -v "$WORK_ROOT:/workspace" \
  -v "$MODEL_ROOT:/models:ro" \
  -v "$PROMPT_ROOT:/data/prompts:ro" \
  -v "$RUN_ROOT:/data/runs" \
  -w /workspace "$IMAGE" bash
```

容器使用的卡必须空闲。16卡机器可把`NPU_IDS`改成实际需要的卡列表；后面的训练/教师分组也要一起修改。`/workspace`和`/data/runs`已挂到主机，退出容器后代码、虚拟环境和结果仍保留。若容器已经创建，后续用`docker start -ai dflash2-27b-train`重新进入，不要重复创建同名容器。`128g`是此示例的共享内存上限，应结合主机内存、workers和缓存规模设置。

#### 第2步：在容器内获取代码、安装训练依赖

以下命令均在**容器内Bash**执行：

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh

cd /workspace
git clone https://github.com/maojunx99/Documents.git
cd /workspace/Documents/qwen38-27b-dflash2/pipeline
export PIPELINE_ROOT="$PWD"

# 继承镜像中已匹配的计算栈，只在独立环境补充训练依赖。
python3 -m venv --system-site-packages /workspace/venvs/dflash2
source /workspace/venvs/dflash2/bin/activate
export PYTHON="$VIRTUAL_ENV/bin/python"
export PY="$PYTHON"
mkdir -p /workspace/environment_records

# 记录并锁住镜像的基础包；解析新增依赖时不允许替换这些包。
python - <<'PY'
import importlib.metadata as md
import json
from pathlib import Path

names = ["torch", "torch-npu", "torchvision", "torchaudio", "vllm",
         "vllm-ascend", "triton-ascend", "numpy"]
versions = {name: md.version(name) for name in names}
assert versions["torch"].split("+")[0] == "2.10.0", versions
assert versions["torch-npu"] == "2.10.0.post4", versions
assert versions["vllm"].split("+")[0] == "0.26.0", versions
assert versions["triton-ascend"] == "3.2.2", versions
out = Path("/workspace/environment_records")
(out / "base-stack.txt").write_text(
    "".join(f"{name}=={version}\n" for name, version in versions.items()))
(out / "base-stack.json").write_text(json.dumps(versions, indent=2))
print(json.dumps(versions, indent=2))
PY

python -m pip install \
  -c /workspace/environment_records/base-stack.txt \
  -r npu/requirements-training.txt
```

`requirements-training.txt`只列新增训练依赖，版本来自已验证环境；它不替代官方计算栈的完整锁定。如果pip报告与基础包冲突，应核对镜像版本和冲突原因，不要删除基础约束后升级整套torch/vLLM。

**不执行`pip install speculators`、`pip install -e npu/source`或安装归档中的`egg-info/requires.txt`。**入口分别加载`npu/source`与`regeneration/source`的适配源码，包中也包含所需发行元数据；旧归档依赖描述含来源路径，不能当作新机器的安装清单。也不要安装通用CUDA版Triton覆盖Triton Ascend。

安装后检查当前解释器、NPU与两个源码快照：

```bash
python - <<'PY'
import importlib.metadata as md
import sys
import torch
import torch_npu
import vllm
import vllm_ascend
import triton

assert torch.npu.is_available()
print("Python:", sys.executable)
for name in ["torch", "torch-npu", "vllm", "vllm-ascend", "triton-ascend",
             "transformers", "datasets", "pyarrow"]:
    print(name, md.version(name))
PY

for snapshot in npu/source regeneration/source; do
  PYTHONPATH="$PWD/$snapshot/src:$PWD/$snapshot/hs_connectors/src:${PYTHONPATH:-}" \
  python -c 'import speculators, hs_connectors; print(speculators.__file__); print(hs_connectors.__file__)'
done
python -m pip freeze > /workspace/environment_records/python-freeze.txt
python check_bundle.py
```

两个快照检查应分别打印本仓库下的源码路径。当前已验证的vLLM-Ascend是提交`155f68974`对应的开发安装，与官方镜像tag并非逐字节相同；上述版本检查和import成功后，仍需实际算子、通信及第4步的完整smoke验证，才能确认新环境可用。

#### 第3步：修改目标模型、数据、并行和采样参数

原始JSONL放在主机的`PROMPT_ROOT`，容器内对应`/data/prompts`。每行包含`messages`或`conversations`，例如：

```json
{"id":"example-001","messages":[{"role":"user","content":"Explain how binary search works."}]}
```

若已有assistant回答，重生成阶段会重新生成它们。纯`input_ids/loss_mask`文件不是重生成题源；只有经过同一target身份核验的生成结果，才可以通过`REGENERATED_DIR`显式跳过重生成。

打开`examples/train/npu_dflash2_27b.sh`，将顶部`INPUT_FILES`数组改成实际题源，例如：

```bash
INPUT_FILES=(/data/prompts/open_perfectblend.jsonl /data/prompts/nemotron_v2.jsonl)
```

其它顶部变量可以修改默认值，也可以用环境变量覆盖。下面是**8卡机器的4张训练卡＋TP2教师**配置：

```bash
export PYTHON=/workspace/venvs/dflash2/bin/python
export TARGET_MODEL=/models/Qwen3.8-27B
# 相同结构的其它权重可改为 /models/Qwen3.6-27B
export FSDP_REPLICAS=1
export FSDP_SHARD_SIZE=4
export TRAIN_DEVICES=0,1,2,3
export TEACHER_TP_GROUPS='4,5'
export REGEN_TP_GROUPS='0,1;2,3;4,5;6,7'
export SAMPLE_MAX_LENGTH=4096 PACKED_TOKENS=4096 MAX_ANCHORS=512
export TEMPERATURE=1.0 TOP_P=0.95 TOP_K=20 PRESENCE_PENALTY=1.5
export REASONING_EFFORT=xhigh
export REGENERATED_DIR=
```

这里重生成可先用全部8卡，生成服务退出后再把0..3用于训练、4/5用于在线教师。16卡机器可以配置2×FSDP4（8张训练卡）并给教师独立分配2/4/8张卡；配置含义和完整参数表见下一小节。改变x/y不会自动调整学习率或保证相同训练轨迹。

#### 第4步：先做小规模验证，再启动正式训练

第一次在新环境运行时，先从自己的原始题源选128条，避免为了测试入口就重生成整套大数据。下面仅写新的smoke输入，不修改原文件；请先把`RAW_SOURCE`改成实际文件：

```bash
export RAW_SOURCE=/data/prompts/open_perfectblend.jsonl
export SMOKE_INPUT_DIR="/data/runs/smoke_inputs_$(date +%Y%m%d_%H%M%S)"
python - <<'PY'
import itertools
import json
import os
from pathlib import Path

out = Path(os.environ["SMOKE_INPUT_DIR"])
out.mkdir(parents=True, exist_ok=False)
with Path(os.environ["RAW_SOURCE"]).open(encoding="utf-8") as src:
    rows = list(itertools.islice((json.loads(line) for line in src if line.strip()), 128))
assert len(rows) == 128, "请选择至少有128条有效对话的原始题源"
with (out / "prompts128.jsonl").open("x", encoding="utf-8") as dst:
    for row in rows:
        dst.write(json.dumps(row, ensure_ascii=False) + "\n")
PY

RUN_DIR="/data/runs/dryrun_$(date +%Y%m%d_%H%M%S)" \
  bash examples/train/npu_dflash2_27b.sh \
  --input "$SMOKE_INPUT_DIR/prompts128.jsonl" --dry-run

RUN_DIR="/data/runs/smoke_$(date +%Y%m%d_%H%M%S)" \
MAX_STEPS=2 EPOCHS=2 TRAIN_RATIO=0.8 WARMUP_RATIO=0 \
GENERATION_MAX_TOKENS=128 GENERATION_CONCURRENCY=4 \
  bash examples/train/npu_dflash2_27b.sh \
  --input "$SMOKE_INPUT_DIR/prompts128.jsonl"
```

末尾的`--input`会覆盖脚本顶部的题源列表，仅用于本次128条检查；`MAX_STEPS=2`限制训练累计更新次数。重生成上限128、80/20划分及warmup0都是smoke设置。根据样本长度，每个小epoch可能包含不同步数，不能把这次检查当作正式训练。入口会先做NPU算子、activation checkpointing、CPU保存和x×y通信/梯度预检，再继续完整流程。

检查smoke运行目录中的`status.json`为`completed`，且`COMPLETION_VERIFICATION.json`中`passed=true`；保留`environment_records`和该运行目录的配置、日志及验证收据。失败时先查看`failure.json`、`stage_processes/*.log`、`audit/failure_rank*.json`和`cleanup_summary.json`，不要复用旧目录覆盖重跑。

smoke通过后，使用**新的输出目录**启动正式五轮训练，不带上面的`--input`覆盖：

```bash
RUN_DIR="/data/runs/qwen27b_full_$(date +%Y%m%d_%H%M%S)" \
MAX_STEPS= EPOCHS=5 TRAIN_RATIO=0.99 WARMUP_RATIO=0.04 \
GENERATION_MAX_TOKENS=8192 GENERATION_CONCURRENCY=34 \
STAGE_TIMEOUT=604800 \
  bash examples/train/npu_dflash2_27b.sh
```

该命令使用顶部`INPUT_FILES`的全部题源。`STAGE_TIMEOUT=604800`为每个阶段7天的示例上限，应按实际数据量和预算调整；输出长度、采样、并发、数据量及卡数会显著影响总耗时。模型检查点和日志保存在挂到主机的`RUN_DIR`，前台Ctrl+C会进入本次owned资源清理。若需要后台运行，另用tmux或已有作业系统承载前台命令。

### 1.2 一键入口的完整参数与验证范围

在`examples/train/npu_dflash2_27b.sh`顶部填写实际路径和设备分配，然后执行：

```bash
bash examples/train/npu_dflash2_27b.sh --dry-run  # 仅打印配置、rank mesh和各阶段命令
bash examples/train/npu_dflash2_27b.sh            # 启动完整流程
```

入口依次完成：模型/tokenizer检查、真实NPU算子和通信预检、目标回答重生成、停止生成服务、原生预处理、启动hidden-state教师与路由、随机初始化草稿并训练、每轮验证与保存、实际回载检查、仅清理本次服务和独占缓存。默认使用所有传入题源，不再自动限为1万条；小规模测试请先准备选定的输入JSONL。

主要可修改项均集中在同一个脚本顶部：

| 变量 | 用途 / 默认值 |
|---|---|
| `TARGET_MODEL`、`INPUT_FILES`、`RUN_DIR` | 完整目标模型、原始对话文件列表、新运行目录 |
| `FSDP_REPLICAS`、`FSDP_SHARD_SIZE`、`TRAIN_DEVICES` | x、y及x×y张训练卡；默认2×4、8..15 |
| `REGEN_TP_GROUPS`、`TEACHER_TP_GROUPS` | 生成阶段及在线教师的TP分组；逗号组内、分号组间 |
| `TEMPERATURE`、`TOP_P`、`TOP_K`、`PRESENCE_PENALTY` | 回答重生成参数；1.0 / .95 / 20 / 1.5 |
| `REASONING_EFFORT`、`GENERATION_MAX_TOKENS` | xhigh / 8192；使用目标原生chat template，客户端不手工拼接system prompt |
| `GENERATION_CONCURRENCY`、`GENERATION_CONTEXT`、`GENERATION_MEMORY` | 每生成服务的并发、上下文及显存预算 |
| `SAMPLE_MAX_LENGTH`、`PACKED_TOKENS`、`MAX_ANCHORS` | 4096 / 4096 / 512；anchors为每rank packed批次共享上限 |
| `EPOCHS`、`LR`、`WARMUP_RATIO`、`SELECTOR_ALPHA` | 5 / 6e-4 / .04 / .1；其余损失与优化器参数也可修改 |
| `MAX_STEPS` | 默认为空；可设为正整数做有界流程测试，表示全任务累计更新上限 |
| `REGENERATED_DIR` | 可选；指定同一目标生成的已核验shards后跳过重生成 |
| `STAGE_TIMEOUT` | 每个工作阶段的时间上限，默认86400秒；大规模重生成/多轮训练需按预算调大 |

例如，仍在对应训练容器内，可用环境变量临时覆盖顶部默认值：

```bash
PYTHON=/path/to/venv/bin/python \
TARGET_MODEL=/models/Qwen3.6-27B \
RUN_DIR=/data/runs/qwen36_scratch_001 \
FSDP_REPLICAS=2 FSDP_SHARD_SIZE=4 \
TRAIN_DEVICES=8,9,10,11,12,13,14,15 \
TEACHER_TP_GROUPS='2,3' \
bash examples/train/npu_dflash2_27b.sh
```

`INPUT_FILES`是脚本顶部的Bash数组，需填写自己的题源文件。这个示例以现有27B结构为蓝本：更换成**相同结构的Qwen3.6-27B权重**时，主要修改`TARGET_MODEL`；预处理、生成服务、在线教师以及冻结的embedding/head都从该目标读取。入口检查hidden size、词表维度、特征层索引、tokenizer ID范围、mask ID往返和原生chat template；模型结构不同则需要明确修改`DRAFT_CONFIG`、`TARGET_LAYERS`、`MASK_TOKEN_ID`并重新验证，不能只改模型名。

换目标权重后应重新生成回答和hidden states。默认全流程自动执行；显式复用旧回答时，要求其`REGENERATION_CONFIG.json`记录相同目标路径及未变化的模型文件身份。旧目录缺少该收据会拒绝复用，不会把其他模型的回答静默当作当前目标数据。原生模板本身可能根据`reasoning_effort=xhigh`渲染reasoning说明，实际token以当前target的模板输出为准。

| 设置 | 训练卡数 | 含义 |
|---|---:|---|
| 1×FSDP4 | 4 | 一组4卡分片 |
| 2×FSDP4 | 8 | 两组4卡分片；相同分片列之间做复制组同步 |
| 2×FSDP2 | 4 | 两组2卡分片 |
| 4×FSDP4 | 16 | 四组4卡分片 |

`TRAIN_DEVICES`数量必须恰好等于x×y。这里每个训练rank都读取不同数据，实际数据并行大小是x×y；修改卡数会改变全局token预算、步数和训练轨迹，不能据此声称与原配方逐位相同。通信预检会在当前拓扑上比较全局批次梯度及检查点回载。

2026-10-19已在910B2C验证本统一入口：128条原始问题→128条新回答→128条prepared记录；2×FSDP4、packed4096、512 anchors、完整词表，各rank2次真实更新，两个小规模epoch的验证与保存、最终模型/AdamW实际回载、复制组状态一致和owned清理通过。该测试将生成上限临时设为128、划分设为80/20、warmup设为0以验证更新与流程，**不是默认五轮配方的质量评测**。另有既有27B 2×FSDP4/20步验证；通用1×FSDP4与2×FSDP2完成了小模型全局梯度/通信/回载检查，不能据此保证所有x/y的完整27B显存容量。Qwen3.6-27B的切换已做命令与结构检查，尚未加载其真实权重运行。

**在线教师还需要独立卡。**默认8张训练卡＋一个TP2教师，共10张；若机器只有8张卡，可以选择4张训练卡＋TP2教师，剩余卡另行分配。不能把“8卡训练”理解为已经包含教师。生成阶段与训练阶段顺序执行，生成卡可以重用；入口确认生成服务退出后才进入训练。

输出中的`COMPLETION_VERIFICATION.json`且`passed=true`表示本次流程、更新计数、验证、回载和owned清理通过；它不代表serving接受长度评测。新目录不可覆盖、失败不自动重训；预检、生成和训练的独立日志、命令、身份与模型/源码清单都会保留。默认五轮保留各轮检查点，应另准备约每轮12GiB以及数据/缓存空间。原生BF16检查点仍不是FP32无损恢复。

## 2. 回答重采样：GPU和NPU均可使用

本流程重新生成目标模型的回答，和训练时教师提供hidden states是两个阶段。历史基础数据的生成参数为：

| 参数 | 数值 |
|---|---|
| 目标模型 | Qwen3.8-27B |
| Temperature / top-p / top-k | 1.0 / 0.95 / 20 |
| Presence penalty | 1.5 |
| Reasoning effort | xhigh |
| 最大输出长度 | 8192 |
| 每个HTTP服务的客户端并发 | 34 |

使用模型原生对话格式。重采样客户端不额外添加system prompt；输入中已有的对话角色交由原生实现处理。原始输出metadata会保留实际采样参数和finish reason。达到输出上限的回答仍保留，训练预处理再应用单独的4096长度上限。生成请求没有另外锁定采样seed，不能据此承诺回答逐字节重现。

**终端A：启动独占生成服务。**按实际空闲卡选择，设备组内是TP，组与组独立。

```bash
# GPU示例：四个TP1服务，端口8500..8503
$PY regeneration/serve.py --backend gpu --model /models/Qwen3.8-27B \
  --device-groups '0;1;2;3' --run /data/runs/regen_servers_001

# NPU示例：另选两个空闲卡，一个TP2服务
$PY regeneration/serve.py --backend npu --model /models/Qwen3.8-27B \
  --device-groups '2,3' --run /data/runs/npu_regen_servers_001
```

上述是两个可选示例，不要同时占用同一批设备。服务启动后输出健康检查结果；前台Ctrl+C只清理该入口启动的进程。生成服务的32768上下文上限是公开入口示例配置，可通过`--max-model-len`调整。

**终端B：对选定题源重采样。**只传入希望使用的来源文件，不会自动下载或混入其它数据。

```bash
$PY regeneration/regenerate.py --model /models/Qwen3.8-27B \
  --input /data/prompts/*.jsonl \
  --endpoints http://127.0.0.1:8500 http://127.0.0.1:8501 \
              http://127.0.0.1:8502 http://127.0.0.1:8503 \
  --run /data/runs/regenerated_001

$PY regeneration/verify_outputs.py \
  --shards /data/runs/regenerated_001/shards --expected-rows 859794
```

如果只起了一个NPU服务，`--endpoints`只填一个地址。859794是历史基础数据的条数，换数据后应填写自己的期望条数。输出`shards/*_out.jsonl`含生成后的`input_ids`、`loss_mask`和metadata，可以直接用于下面两条训练路线。入口对不同题源的ID加前缀，避免来源之间的ID碰撞。

预先查看命令而不生成任何内容：在`serve.py`或`regenerate.py`命令末尾加`--dry-run`。

## 3. GPU从零训练

```bash
$PY gpu/run_pipeline.py \
  --model /models/Qwen3.8-27B \
  --data-dir /data/runs/regenerated_001/shards \
  --run /data/runs/gpu_scratch_001 \
  --teacher-devices 0,1,2,3 --train-devices 4,5,6,7
```

入口会先拒绝已有输出目录和被占用的设备，然后进行原生预处理、启动4个TP1教师和HTTP路由、启动DDP4训练。每轮原生验证并保存检查点；无旧草稿初始化、无旧优化器自动恢复。

固定基础参数：随机草稿、5层、taps 6/20/34/48/62、block8/spec7、selector rank256/top16、CE0.1＋TV0.9、selector CE权重0.1、gamma4、AdamW6e-4/WD0、cosine/warmup4%、seed42、5轮、单样本4096、packed8192、1024 anchors、99/1、每rank 12 workers/prefetch4。anchors是每卡packed批次共享的上限。

先做有界smoke可加`--max-steps 3`并使用另一个新输出目录；学习率调度仍保留完整训练计划，smoke不会成为后续从零训练的初始化。查看完整argv可加`--dry-run`。

输出包括`PIPELINE_CONFIG.json`、源码hash、各进程身份和命令、教师provenance、trainer.log、原生轮末验证、checkpoints及cleanup。任何失败保留原输出，不自动重训。退出时只处理本次启动且PID/starttime/命令仍匹配的进程。

## 4. NPU FSDP4＋512 anchors完整1万条流程

先设置路径；不需要固定原机器的用户名或旧归档目录：

```bash
export DFLASH_PYTHON="$PY"
export DFLASH_TARGET_MODEL=/models/Qwen3.8-27B
export DFLASH_REGENERATED_DATA=/data/runs/regenerated_001/shards
export DFLASH_TRAIN_DEVICES=8,9,10,11

# 新目录中进行实际算子/activation-checkpoint/CPU保存/FSDP检查
$PY npu/scripts/prepare_preflights.py --train-devices "$DFLASH_TRAIN_DEVICES"

# 必须使用本管线npu/runs下的一个新名字
B="$(pwd)/npu"
$PY "$B/scripts/run_pipeline_fsdp.py" --run "$B/runs/train_001" \
  --train-devices "$DFLASH_TRAIN_DEVICES" --teacher-devices 2,3 --teacher-port 8352

# 上一个前台监督器退出后再独立验证、打包轻量代码和证据
$PY "$B/scripts/finalize_after_exit.py" --run "$B/runs/train_001"
```

默认训练8/9/10/11、教师2/3来自已验证的同平面HCCL拓扑。其它910B机器要根据实际拓扑选择四张可互通训练卡；不应仅凭卡号相邻就认为跨平面可用。入口会重新做全训练拓扑HCCL检查。

NPU入口会从已有重采样结果按基础八类来源的比例选取10000条、去重并生成manifest，重新prepare4096、99/1划分，然后启动TP2教师和FSDP4训练。锁定512 anchors、完整词表、CE/TV、FP32 master与AdamW状态、BF16计算；启用非重入activation checkpointing、前向前zero_grad和SDPA。

验证通过需要四rank真实更新、有限loss/梯度、原生验证、实际模型与优化器回载、源码/输入指纹一致及owned资源释放。`COMPLETION_VERIFICATION.json`中的`passed=true`才表示成功。新数据的实际步数由DataLoader决定；历史10000条为每rank909步。

原生落盘检查点为BF16，模型、AdamW moments和浮点step会舍入，而且缺少RNG快照；保存/回载通过不表示FP32无损恢复。GPU原生基础训练也有该序列化限制。后续实验的FP32 master检查点代码独立保留在`gpu/continuation/`。

## 5. GPU后续训练代码

`gpu/continuation/`保存文章最终模型对应的补充数据训练实现，包括固定token预算/梯度累积、成对输入与anchor记录、共享特征缓存、分布匹配目标、FP32 master优化器、检查点导出、数据选择与补充回答生成。

主要入口是`run_targeted_main_7gpu_v2.py`、`train_targeted_arm_v3.py`和`finalize_targeted_data_v3.py`。`reference_protocol/`记录原固定预算与参数，`source_integration_v2/`和`reference/`是必要实现依赖。

这组入口是锁定实验的源码归档，依赖既有起点检查点、实际数据、锁定的schedule、预检收据和hash清单。公开协议中的机器路径已改为`/workspace`和`/models`占位；不能原样执行旧协议，也不能把它当作随机初始化配方。换路径或数据后需要重新构建并核验锁定清单；不要删除原有hash断言或复用旧检查点伪装新实验。对应实际算法实现已全部保留，不用仅凭“overlap”环境变量猜测训练目标。

## 6. 检查与来源

```bash
$PY check_bundle.py
```

该检查不导入torch，不使用GPU/NPU，也不连接服务器。它核验文件hash、Python语法、公共入口命令规划和CPU数据结构。硬件环境、算子及训练质量仍需要在目标设备上确认。

源码版本、修改文件和验证边界见[PROVENANCE.md](PROVENANCE.md)；原始上游Apache-2.0许可证保留在各源码快照。发布目录不包含账户密码、SSH配置、私钥、模型权重或原始问题/回答。
