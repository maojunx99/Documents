# GPU：回答重采样与从零训练

以下命令在已配置的Linux GPU环境中，从`pipeline/`目录执行。`PY`指向该环境的Python。

已有训练环境记录：PyTorch2.13.0、vLLM0.29.0、Transformers5.16.1、hs-connectors0.7.1a0；[完整包版本](environment-recorded.json)是运行记录，不是通用安装锁文件。Speculators与hs-connectors的适配源码已保留在`gpu/source/`，入口通过`PYTHONPATH`优先加载；其余依赖需由GPU环境提供。不要直接以PyPI最新包替换这些适配。

基础训练已在H200完成；这里的公开编排入口只做过静态和CPU检查，尚未重跑GPU硬件验证。新机器应先检查环境并做有界试跑。

## 1. 准备输入

- 完整Qwen3.8-27B目标目录，包含config、tokenizer和所有权重分片。
- 原始问题/对话JSONL，每行含`messages`或`conversations`。
- 可写的数据、hidden缓存和检查点空间；每次使用新的输出目录。

若已有本目标模型生成的`*_out.jsonl`，可直接进入第3步。

## 2. 回答重采样

终端A：按实际空闲卡启动四个TP1生成服务，端口8500～8503。

```bash
$PY regeneration/serve.py --backend gpu --model /models/Qwen3.8-27B \
  --device-groups '0;1;2;3' --run /data/runs/regen_servers_001
```

终端B：对选定题源生成回答。

```bash
$PY regeneration/regenerate.py --model /models/Qwen3.8-27B \
  --input /data/prompts/*.jsonl \
  --endpoints http://127.0.0.1:8500 http://127.0.0.1:8501 \
              http://127.0.0.1:8502 http://127.0.0.1:8503 \
  --run /data/runs/regenerated_001
$PY regeneration/verify_outputs.py --shards /data/runs/regenerated_001/shards
```

默认重生成参数为T1、top-p.95、top-k20、presence1.5、xhigh、max_tokens8192、每服务客户端并发34；使用模型原生对话格式，客户端不额外添加system prompt。需要改变参数时使用`regenerate.py --help`。默认不锁定生成seed，不能保证回答逐字节复现。

输出`shards/*_out.jsonl`含`input_ids`、`loss_mask`、采样参数与finish reason。达到输出上限的回答仍保留，训练预处理另应用4096长度上限。生成完成后在终端A按Ctrl+C，确认生成服务清理、设备释放后再训练。命令加`--dry-run`可只查看计划。

## 3. 从零训练

```bash
$PY gpu/run_pipeline.py \
  --model /models/Qwen3.8-27B \
  --data-dir /data/runs/regenerated_001/shards \
  --run /data/runs/gpu_scratch_001 \
  --teacher-devices 0,1,2,3 --train-devices 4,5,6,7
```

入口执行预处理、4个TP1在线教师及路由、DDP4训练、每轮原生验证/检查点与进程清理；拒绝已有输出目录和被占用设备。草稿随机初始化，不自动加载旧草稿或优化器。

| 参数 | 基础配方 |
|---|---|
| 结构 | 5层；特征层6/20/34/48/62；block8/spec7；selector rank256/top16 |
| 目标 | CE.1＋TV.9；selector CE权重.1；fixed-exp-decay gamma4 |
| 优化器 | AdamW6e-4、WD0、cosine、warmup4%、seed42 |
| 数据预算 | 单样本4096、packed8192、1024 anchors、99/1划分 |
| 训练 | 5轮、DDP4、每rank12 workers/prefetch4 |

anchors是每rank packed批次共享上限。入口加`--max-steps 3`可做有界smoke，仍保留完整五轮学习率计划；加`--dry-run`查看完整命令。原生BF16落盘与缺少RNG快照，不表示FP32无损恢复。

输出包括配置、命令/进程身份、教师provenance、trainer.log、验证、checkpoints与cleanup。任何失败均保留现场，不自动重训；清理只处理本次启动且身份匹配的进程。

## 4. 后续训练代码

[continuation/](continuation/)保存文章最终模型对应的补充训练实现，包含固定token预算/梯度累积、配对输入与anchor记录、共享特征缓存、分布匹配目标、FP32 master优化器和数据选择。

主要入口为`run_targeted_main_7gpu_v2.py`、`train_targeted_arm_v3.py`、`finalize_targeted_data_v3.py`。它们是锁定实验的源码归档，依赖既有起点、真实数据、schedule及hash清单；不能直接作为从零训练入口。换路径或数据应重新构建清单，保留原有断言；协议中的`/workspace`、`/models`为占位路径。

[验证记录](../VALIDATION.json) · [源码说明](../PROVENANCE.md) · [返回平台选择](../README.md)
