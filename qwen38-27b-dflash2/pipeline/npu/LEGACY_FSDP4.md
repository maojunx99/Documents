# 原1万条FSDP4验证入口

这是已归档的固定1万条流程。新任务使用[NPU运行说明](README.md)中的分步脚本。以下命令仍从`pipeline/`执行，`PY`为已配置的NPU Python。


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
