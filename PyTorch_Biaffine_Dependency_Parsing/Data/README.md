# Data 目录说明

本目录是项目唯一的数据根。所有 conll 语料 + 预训练词向量都放这里。

## 当前内容

```
Data/
├── ctb51/                  ← HIT 依存语料(任务用,代码默认读这里)
│   ├── train.conll         ← 8 301 句, 9.3 MB,md5=d82e6679...
│   ├── dev.conll           ← 534 句, 589 KB,md5=b71b08dc...
│   └── sample.conll        ← 1 句, 0.3 KB,本地冒烟测试用
├── thu/                    ← 清华语义依存语料(备用,标注体系不同)
│   ├── train.conll         ← 5.5 MB,语义角色标签如"核心成分/受事/限定"
│   └── dev.conll           ← 579 KB
├── Embed/
│   ├── giga.100.txt.sample ← 100 行 100d 词向量样例(冒烟用,真实训练 OOV 97%)
│   └── README.md           ← 完整词向量下载 / 命名 / 使用说明
└── README.md               ← 本文件
```

## 数据集来源

> 第二届 NLP&CC 2013 技术评测样例:
> http://tcci.ccf.org.cn/conference/2013/pages/page04_sam.html
> 下载地址:http://tcci.ccf.org.cn/conference/2013/dldoc/evsam05.zip

CONLL 标注格式 10 列(第 9/10 列本数据未用):
```
ID  FORM  LEMMA  CPOSTAG  POSTAG  FEATS  HEAD  DEPREL  PHEAD  PDEPREL
```
- `HEAD`:当前词的中心词序号(0 表 ROOT)
- `DEPREL`:依存关系类型(HIT)或语义角色(THU)

## ctb51 vs thu

| 项 | ctb51 (HIT) | thu |
|---|---|---|
| 标注体系 | **句法依存** | **语义依存** |
| 标签举例 | `SUB`, `OBJ`, `VMOD`, `NMOD`, `ROOT` | `核心成分`, `受事`, `限定`, `方式` |
| 句子数 | train 8 301 / dev 534 | train ~ / dev ~ |
| 任务关系 | **本实验主用** | 备用,**不能直接跟 HIT 做对比**(标签集完全不同) |

> 历史:目录名 ctb51 来自原作者最初配置,实际放的是 HIT 语料(md5 与 NLP&CC 2013 HIT 一致);为不破坏代码默认路径,目录名保留。

## 测试集去哪了

任务说明明确"测试集按惯例不公开"。代码中:
```ini
[Data]
test_file = ./Data/ctb51/dev.conll   # 故意与 dev 同
```
所以训练日志里"Test UAS"实际是 dev UAS。**不影响对比公平性**(所有对比都在同一份 dev 上),但报告里要写明。

## 添加新数据时

1. 新增 conll 数据:在 `Data/` 下建子目录(如 `Data/ctb7/`),配置文件用 `--override Data.train_file=./Data/ctb7/train.conll` 切过去。
2. 新增词向量:放 `Data/Embed/`,文件名按 `Embed/README.md` 规约。
3. 不要把数据放在 `Data/` 之外的位置。
