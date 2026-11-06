# 多机器在线teacher

trainer仍在一台机器上运行；在线teacher可以分布在多台机器，每个远端服务使用一组独立TP卡。本机teacher和远端teacher可以混用，也可以只使用远端teacher。NPU的x×FSDPy、GPU的DDP4训练参数均保持原有设置；这不是多机trainer/FSDP功能。

本功能只负责**训练时的hidden states提取**。回答重生成仍使用原有重生成入口，五基准仍使用评测入口，不能把普通文本生成服务当作在线teacher。

## 1. 准备各台机器

- 各机器使用同一版本的本管线与对应平台依赖。当前支持NPU trainer接NPU teacher，或GPU trainer接GPU teacher；没有开放混合硬件teacher。
- 每台teacher准备相同的完整目标权重和tokenizer。挂载路径可以不同，接入时会逐文件核对完整SHA256，不只比模型名称或文件大小。首次接入会读取全部目标权重，需预留校验时间。
- trainer和teacher必须挂载**同一共享文件系统、同一绝对路径**，如`/shared/dflash2`。双方需要创建、读取和删除特征文件的权限，注意容器UID/GID或共享组权限。每个teacher使用独立、全新的任务子目录，不能复用旧任务。也不能把各机器互不共享的`/dev/shm`当成共享目录。
- trainer能访问teacher网关的私网端口，或通过用户自行建立的SSH隧道访问。原生vLLM端口仅绑定teacher本机loopback；不要开放到公网。
- 使用同一个私有随机API密钥文件认证。仓库不提供密码、SSH账户或密钥值，也不会自动SSH登录其他机器。

在`pipeline/`目录执行一次：

```bash
bash examples/teachers/create_key.sh
```

默认写入`/run/secrets/dflash2_teacher.key`，拒绝覆盖。通过安全渠道把同一密钥文件放到trainer和所选teacher机器，权限设为`0600`。可在脚本顶部或环境变量中修改`KEY_FILE`、`PYTHON`。不要把密钥文件提交Git。

## 2. 在每台teacher机器启动一个服务

修改[run_teacher.sh](../examples/teachers/run_teacher.sh)顶部变量，或者通过环境变量传入：

| 参数 | 示例 / 含义 |
|---|---|
| `BACKEND` | `npu`或`gpu`；与trainer平台一致 |
| `PYTHON` | 对应训练/teacher环境的Python |
| `TARGET_MODEL` | 这台机器上的目标模型目录 |
| `TEACHER_DEVICES` | `0,1`表示一个TP2；`0`表示TP1；卡号须空闲 |
| `SHARED_RUN` | `/shared/dflash2/teacher_a_job01`，新目录，在trainer上同路径可见 |
| `BIND_HOST` | teacher的可达私网IP；默认loopback，适合SSH隧道 |
| `GATEWAY_PORT` | `8395`，trainer访问的认证网关端口 |
| `NATIVE_PORT`、`DISTRIBUTED_PORT` | `8394`、`29300`；同机起多个实例时分别改为独立端口 |
| `API_KEY_FILE` | 本机的私有认证文件路径 |
| `TEACHER_CONTEXT` | 默认16384，至少大于trainer的packed请求长度，需容纳额外生成token |
| `TEACHER_MAX_NUM_SEQS`、`TEACHER_MEMORY` | 默认4、0.8 |
| `LEASE_SECONDS` | 默认120秒；trainer定时续租，失联后释放本服务的teacher |

```bash
bash examples/teachers/run_teacher.sh --dry-run
bash examples/teachers/run_teacher.sh
```

NPU脚本会加载CANN/ATB环境。服务先等待trainer接入，**在身份核验、租约认领成功后才检查空卡并启动原生teacher**；等待期间不占加速卡。默认最多等待24小时，不会提前启动GPU/NPU模型服务。

其他teacher机器重复这一步，使用各自的IP、卡组和独立`SHARED_RUN`。同一机器也可起多组TP服务，但卡组、端口和任务目录须各自独立。teacher任务为一次性：完成、失败或过期后不自动重新启动，用新的任务目录重新发起。

## 3. 在trainer机器填写远端列表

复制[remote_teachers.example.json](../examples/teachers/remote_teachers.example.json)到自己的配置目录，修改每个teacher的四项字段：

| 字段 | 含义 |
|---|---|
| `name` | 唯一名称，例如`teacher_a` |
| `url` | 网关base URL，例如`http://teacher-a.internal:8395`；不加`/v1` |
| `api_key_file` | **trainer机器上**的密钥文件路径；相对路径按JSON文件位置解析 |
| `shared_root` | 对应teacher的`SHARED_RUN`，必须与teacher端绝对路径完全相同 |

可填写任意多台固定teacher，不限于示例中的两个。不要将用户名、密码或认证值写到URL中。

### NPU：一键与分阶段入口

在`config.local.npu.sh`中设置：

```bash
REMOTE_TEACHERS=/workspace/configs/remote_teachers.json
TEACHER_TP_GROUPS=""  # 仅远端teacher；保留"4,5"则是本地TP2＋远端混用
```

之后沿用原命令：

```bash
bash examples/npu/04_run.sh --dry-run
bash examples/npu/04_run.sh --smoke
bash examples/npu/04_run.sh
```

smoke和正式训练分别使用新的本地任务、远端任务目录，需要按第2步重新启动各teacher服务。分阶段运行仍用原来的五个脚本，`04_train.sh`才会认领远端teacher；前面预检、重生成和预处理阶段不会占用远端加速卡。所有阶段继续使用同一份配置。

### GPU：基础训练入口

```bash
python gpu/run_pipeline.py --model /models/Qwen3.8-27B \
  --data-dir /data/answers/shards --run /data/runs/remote_train_01 \
  --train-devices 0,1,2,3 --teacher-devices "" \
  --remote-teachers /workspace/configs/remote_teachers.json --dry-run
```

正式运行移除`--dry-run`。GPU trainer仍是4卡DDP；`--teacher-devices`填写本机TP1教师卡，留空则仅使用远端教师。远端服务的TP大小由各teacher的`TEACHER_DEVICES`决定。回答生成与预处理仍使用原来的数据管线。

## 4. 接入检查、文件传输和退出

接入前核对目标权重/tokenizer完整SHA、该平台Speculators/hs-connectors的Python源码SHA、torch/Transformers/vLLM及NPU扩展版本、特征层、hidden宽度、共享目录和teacher上下文容量。然后运行32/256-token真实hidden提取探针，核对BF16、形状、有限值和token对应关系；包版本与源码一致也不等于二进制或BF16结果逐字节确定。

HTTP请求中的token IDs原样转发。原生teacher写入自己的`staging/`，同机网关等待写锁结束，校验完整safetensors字节范围与请求token IDs，再`fsync`并原子重命名到`ready/`。trainer只收到已完成的`ready/`文件，读取后使用原有FileTransfer删除它。**不使用跨主机flock协调读写**，也不通过HTTP发送完整hidden tensor。共享存储需要足够带宽，增加teacher不保证训练吞吐线性增加。

trainer保持租约心跳；发现远端失联或服务退出时，当前公开入口保留失败证据并停止本次训练，不静默丢样本、替换模型或自动重训。当前为固定teacher池，没有复刻历史运行中热加入/自动本地回退功能。

正常退出顺序：先停止本机trainer/router等消费者，再通知远端服务停止其独占teacher，复核目标/源码指纹并清理其`staging/ready`。仅在远端清理收据通过后写训练完成。远端进程操作使用PID、starttime、命令和私有session核验，不使用全局pkill。

trainer失联导致租约过期时，远端停止自己的teacher，**保留特征文件**，避免仍活着的trainer读到被删除的文件。独立CPU guardian还会在网关owner异常退出时清理已登记的teacher进程；特征文件仍保留。异常现场不得直接复用或递归删除，应先核实所有消费者退出，再清理该任务独占的缓存。

主要证据：trainer的`remote_teachers/ADMISSION.json`、各teacher接入探针、`ROUTER_BACKENDS.json`、`router_routing.jsonl`和`remote_teachers/CLEANUP.json`；远端的`CONTRACT.json`、`CLAIM.json`、`teacher_provenance.json`、`environment.json`、`routing.jsonl`、`CLEANUP.json`及`GUARDIAN_CLEANUP.json`。这些文件不保存API密钥值。

## 5. 验证范围

2026-10-19曾在253的trainer＋本地3组TP2、243的额外1组TP2上通过实际训练请求验证，使用共享CFS和同机写完再发布的私有集成版本。

本次是该方案的通用公开编排，增加了可配置endpoint、认证、固定池接入、租约和独立guardian。新增CPU测试使用模拟原生teacher，覆盖两teacher并发HTTP传输、文件完整性与token对齐、错误身份/认证拒绝、租约过期保留缓存，以及正常清理。**历史硬件通过不等于本次公开入口已完成新的多机训练验证**；新部署应先用上述`--smoke`运行完整小步测试。
