# Qwen3.8-27B PB860K 重采样训练数据

[下载完整数据包（GitHub Release）](https://github.com/maojunx99/Documents/releases/tag/pb860k-regenerated-20261022)

这是27B训练实际使用的重采样原始数据，共 **859,794条、145个JSONL分片**。保留原文件的 `id`、`primary_id`、`input_ids`、`loss_mask`、`text` 和 `metadata`，没有为了上传而删除字段、截断或重新采样。ID已加数据域前缀。

原文件合计31,766,686,735 bytes（31.77GB / 29.59GiB）；gzip压缩后合计 **8,369,601,216 bytes（8.37GB / 7.79GiB）**，共20个独立 `tar.gz` 包。文件体积较大，是因为文本、token ID和逐token mask同时保存。各包约0.29–0.49GB，可单独下载、重试及解压。

大文件保存在同一仓库的Release附件中，避免将数GB二进制文件写进Git历史。本目录提供固定版本的清单、下载脚本和说明。

## 下载及校验

仅依赖Python标准库，支持Linux和Windows：

```bash
python qwen38-27b-dflash2/data/download_pb860k.py --output ./pb860k --workers 4
```

脚本会校验清单SHA256、每个压缩包SHA256，以及解压后每个文件的SHA256。已有且校验通过的文件会复用，未完成的下载可续传。最终训练源目录为 `./pb860k/regen_data/`，训练输入为其中的 `*_out.jsonl`。

下载并保留压缩包、同时解压，需预留约41GB空间，建议至少45GB。只下载压缩包可用：

```bash
python qwen38-27b-dflash2/data/download_pb860k.py --output ./pb860k --download-only
```

也可以从Release下载全部20个包和 `SHA256SUMS`，在Linux下手动校验及解压：

```bash
sha256sum -c SHA256SUMS
mkdir -p pb860k
for f in pb860k-regen-*-of-20.tar.gz; do tar -xzf "$f" -C pb860k; done
```

**每个包都可独立解压，不要将这些包拼接成一个文件。** 全部解压到同一目录即可得到完整的 `regen_data/`。

## 数据构成

| 来源 | 重采样记录数 |
|---|---:|
| AutoIF | 37,389 |
| Evol-CodeAlpaca | 107,789 |
| LMSYS | 49,002 |
| MetaMathQA | 141,738 |
| Orca Math | 183,725 |
| UltraChat | 201,774 |
| UltraFeedback | 59,794 |
| UltraInteract | 78,583 |
| 合计 | **859,794** |

保留包括 `finish_reason=length` 在内的原始重采样记录；生成参数和完成原因见各条记录的 `metadata`。数据包同时保留原始 `MANIFEST_REKEYED.txt` 和生成阶段的 `.errors.jsonl` 旁路日志；旁路日志不是训练输入。

## 与预处理训练数据的区别

本次发布的是 **`regen_data` 原始重采样数据**，没有附带Arrow缓存。原训练的 `prepared` 为859,613条、52个Arrow分片，单样本预处理上限4096 token，训练packed预算8192 token/rank。重新预处理时应使用这里的原始JSONL；不要将原始记录数和prepared记录数混为一谈。

## 完整性与来源

- 导出源：H200训练目录 `qwen38_27b_pb860k_20260926/regen_data/`。
- 145个训练分片的SHA256、字节数和记录数均与历史 `MANIFEST_REKEYED.txt` 一致。
- 所有20个压缩包均已完整解压读取，逐文件SHA256复核通过。
- 打包期间源文件字节数和修改时间保持不变。
- [逐文件及逐包清单](pb860k_regenerated.manifest.json)、[压缩包SHA256](pb860k_regenerated.SHA256SUMS)；Release中也附带相同清单和打包脚本。

数据源各自的许可和使用条件仍适用；本次打包不额外授予上游数据的许可。
