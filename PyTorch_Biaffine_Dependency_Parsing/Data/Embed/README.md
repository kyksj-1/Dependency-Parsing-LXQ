# Embed 目录说明 — 预训练词向量放这里

## 当前内容

- `giga.100.txt.sample` — **100 行 100d 词向量样例**,只够冒烟测试。真实训练在本数据上 OOV ≈ 97%,UAS 没意义。

## 真实实验需要下载的文件

来源:**[Chinese-Word-Vectors](https://github.com/Embedding/Chinese-Word-Vectors)**(北师大开源,Skip-Gram with Negative Sampling)

按 MISSION 要求做"维度对比 + 领域对比",**最少下载 4 个文件**:

| 编号 | 推荐文件 | 维度 | 领域 | 用途 |
|---|---|---|---|---|
| ① | `sgns.baidubaike.bigram-char` | 300 | 百度百科 | 维度对比 + 领域对比基线 |
| ② | (百度百科 100d 版本) | 100 | 百度百科 | **维度对比** |
| ③ | `sgns.renmin.bigram-char` | 300 | 人民日报 | **领域对比** |
| ④ | `sgns.weibo.bigram-char` | 300 | 微博 | **领域对比** |

单个文件压缩后 800 MB ~ 2 GB,**先开始下**。

## 下载方式

### 方式 1:本地下载后 scp 到服务器(推荐)

下载 → 解压 → scp 到服务器对应目录,避免服务器外网带宽问题。

### 方式 2:服务器直连下载

Chinese-Word-Vectors 用百度云盘,服务器直连困难。可以走 Google Drive / Hugging Face 镜像:
- 部分镜像在 https://huggingface.co/datasets 上(搜 chinese-word-vectors)

## 放置位置 + 配置切换

下载后**直接放在本目录** `Data/Embed/`,然后命令行 override 切换:

```bash
# 维度对比 A:100d 百度百科
python main.py --device cuda:0 --tag dim100_baike \
    --override Model.embed_dim=100 \
               Embed.pretrained_embed_file=./Data/Embed/sgns.baidubaike.bigram-char.100d

# 维度对比 B:300d 百度百科
python main.py --device cuda:0 --tag dim300_baike \
    --override Model.embed_dim=300 \
               Embed.pretrained_embed_file=./Data/Embed/sgns.baidubaike.bigram-char

# 领域对比:300d 人民日报
python main.py --device cuda:0 --tag dim300_renmin \
    --override Model.embed_dim=300 \
               Embed.pretrained_embed_file=./Data/Embed/sgns.renmin.bigram-char
```

⚠️ **`embed_dim` 必须与文件维度一致**,否则 `nn.Embedding` 加载时会报 shape 不匹配。

## 文件首行注意事项

word2vec 文本格式首行可能是 `vocab_size dim`(如 `364990 300`),`Embed.py` 实测会自动跳过(碰到 `len(values)<=3` 的行就 skip)。
若手动检查发现下载文件首行没数字头(直接是 `<word> v1 v2 ... vN`),也能跑。

## 命名规约

为了让 `--tag` 一眼识别,推荐命名:`sgns.<domain>.<features>.<dim>d`
- domain:`baidubaike` / `renmin` / `weibo` / `sogou` / ...
- features:Chinese-Word-Vectors 提供的 `bigram-char` / `word` / `char` 等
- dim:`100` / `300`

例:`sgns.renmin.bigram-char.300d`
