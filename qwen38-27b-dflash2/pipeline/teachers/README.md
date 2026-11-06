# 多机器在线教师：原生服务＋共享盘

适用于自用、可信私网训练。**没有认证令牌、create_key、独立HTTP网关或租约服务。** trainer仍在单机运行；教师可以分布到其他NPU/GPU主机。同平台组合，训练计算和loss保持原有配置。

## NPU：顶层配置即可

在`config.local.npu.sh`中填写：

```bash
TEACHER_NODES=(
  "local|4,5|2"
  "10.0.0.12|8,9,10,11|2|existing-container"
)
TEACHER_SHARED_ROOT=/mnt/shared/dflash2
TEACHER_SSH_USER=your_user
```

每行是**机器/SSH别名｜物理卡号｜TP｜现有容器名（可选）**。示例为本机一组TP2、远端两组TP2；`TEACHER_NODES`非空时替代`TEACHER_TP_GROUPS`。远端默认Python、模型、容器名可分别改`TEACHER_PYTHON`、`TEACHER_MODEL`、`TEACHER_CONTAINER`。教师通过CANN查询物理卡到容器逻辑卡的映射。

后续仍调用`examples/npu/04_run.sh`或分阶段脚本。脚本复用容器和依赖，快照小体积代码后直接运行原生vLLM teacher；不会另拉镜像或修改SSH凭据。训练容器的SSH需已能非交互连接目标主机，并有权限docker exec已有容器。

| 通道 | 作用 |
|---|---|
| HTTP | trainer原有router分发token-ID请求到原生teacher，返回文件句柄 |
| CFS等共享盘 | teacher保存完整hidden states，trainer直接读取后删除 |
| SSH | 仅启动、检查、停止本次远端进程；不读取hidden states |

各容器须具备相同的管线/依赖和目标权重，挂载同一个可读写共享目录、相同绝对路径。自动模式核对目标文件大小/mtime及小配置SHA、共享文件读写探针，然后做32/256-token真实BF16/有限值/token对应检查。这不是跨机器hidden值逐字节确定性保证。

`TEACHER_SHARED_ROOT`存放本次代码、身份及共享特征。自动生成独占的`HIDDEN_CACHE`；手动指定时同样必须是各节点共享的新目录，不能使用各机独立的`/dev/shm`。端口从`TEACHER_NODE_PORT`起，TP组间隔200；HTTP与分布式端口都需可用。

## 原生共享文件传输

使用`shared_file`后端，复用vLLM原有hidden提取、异步设备到主机复制和writer线程，只改变文件发布步骤：写临时文件→fsync→同目录原子rename→trainer读取最终文件。写失败有明确标记，不把未写完文件交给训练，也不依赖跨主机flock。没有额外的teacher HTTP代理。

普通`file`后端仍保留，适合原有单机路径。Mooncake源码后端也仍保留；旧GPU链路曾使用Mooncake，但旧NPU入口实际是file。这版先完成共享盘路径，没有将未验证的NPU Mooncake当作默认，也不会自动静默切换传输方式。

## 预启动原生服务

已有平台管理的teacher可直接HTTP＋共享盘接入，不需要trainer SSH。教师用`examples/teachers/run_teacher.sh`启动原生`shared_file`服务，再填写[示例JSON](../examples/teachers/remote_teachers.example.json)：只需`name`、`url`、`shared_root`。trainer设置`REMOTE_TEACHERS`且本地教师组可留空；GPU入口也可传`--remote-teachers`。

预启动模式的所有`shared_root`须对应同一个独占缓存：NPU把`HIDDEN_CACHE`设为该路径；GPU使用共享盘上的`RUN_DIR/hidden_states`。

这些预启动服务的生命周期由其原owner管理，管线不会按URL杀服务。推荐NPU日常使用上面的`TEACHER_NODES`，自动管理本次独占教师；不要与`REMOTE_TEACHERS`同时设置。

## 退出和验证边界

正常退出先停trainer/router，再按PID/starttime/私有session停止本次启动的native教师及后代，最后清理独占缓存。失败保留身份、命令和日志，不自动重训。跨机断连时可能无法确认远端退出，清理必须报失败，不能推断卡已释放；没有新增守护服务。

主要证据：`teacher_deployment/JOB.json`、共享job中`node_*_control/identity.json`与`STOPPED.json`、各教师`TARGET_IDENTITY.json`、`DEVICE_MAPPING.json`、`teacher_provenance.json`，以及`remote_teachers/ADMISSION.json`和`router_routing.jsonl`。

最新硬件与CPU验证范围以[VALIDATION.json](../VALIDATION.json)为准；历史253＋243训练通过不替代新入口实际验证。
