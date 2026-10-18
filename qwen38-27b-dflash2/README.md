# Qwen3.8-27B DFlash2训练技术博客

主稿：[Qwen3.8-27B_DFlash2_训练实践.md](Qwen3.8-27B_DFlash2_训练实践.md)。

发布或移动文档时，请同时保留`figures/`目录。正文嵌入PNG，每张图另附SVG矢量版本；公式采用常见Markdown数学语法。

文章以从零训练的数据配比、模型结构和训练参数为主体。第三节集中说明参数选择，并纳入从头重训第一轮的损失权重对照；第六节保留固定已有模型后重跑一轮、补充数据和概率分布匹配等续训探索。五基准主表的接受长度、生成速度和条件接受率统一对应同一最终模型；DSpark为历史测试参考。

配图分为两类：

- DFlash2机制图：引自[Inco AI官方博客](https://inco.ai/blog/dflash2/)的图1和图4，存放在`figures/official/`，正文图注保留出处。网页内嵌SVG导出为静态SVG与高分辨率PNG；候选路径图取动态图的一帧，字体替换为兼容的Arial，图形、标签、配色和连线保留。来源信息见`figures/official/SOURCES.json`。
- 训练与评测图：七张图由`data/blog_data.json`和`data/first_epoch_ablations.json`中的数值绘制。基础训练曲线、参数对照、补充训练曲线和最终模型推理评测在图注中分别说明。

`data/evaluation_identity.json`保留评测模型标识、采样参数、结果来源及核对结果。本次从归档记录重算了官方与本文模型在两种温度下的640次请求，确认主表与配图的接受长度、速度、条件接受率和截断数量一致。没有新增训练或推理实验。

参数消融另从归档记录重算960次请求，配置与来源见`data/first_epoch_ablations.json`。从随机初始化重训和从已有模型续训分别列出，均未混入最终模型的主评测表。

如需重画训练与评测图，安装`matplotlib`与`numpy`后运行：

```powershell
python -X utf8 render_figures.py
```

旧版文稿保存在本地`revisions/`中，不包含在当前发布包内。发布包不包含模型权重和原始题目、回答文本。
