# Qwen3.8-27B DFlash2训练管线

按运行设备选择文档：

| 平台 | 运行说明 | 入口 |
|---|---|---|
| GPU | [GPU训练与重采样](gpu/README.md) | `gpu/run_pipeline.py` |
| NPU | [NPU分步运行](npu/README.md) | `examples/npu/01_container.sh`～`04_run.sh` |

两条路线都包含目标回答重生成、数据预处理、在线教师、草稿训练、原生验证和检查点。模型权重、原始题源、设备驱动需另行准备；Speculators及hs-connectors的适配源码随仓库提供。

NPU支持可配置的x×FSDPy；GPU保留4张训练卡DDP4＋4个TP1教师的基础配方。两条路线的预算与验证范围不同，详见各自文档。

- [验证记录](VALIDATION.json)：区分硬件实测、CPU检查和尚未验证的范围。
- [源码与修改说明](PROVENANCE.md)：版本、适配范围及许可证来源。
- `check_bundle.py`：无设备的文件完整性、语法和命令规划检查。
