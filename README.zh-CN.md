# Dependency Parsing

[English README](README.md)

本仓库是一个中文依存句法分析课程项目，核心模型基于 Dozat 和 Manning 的 Deep Biaffine Parser。我们从一个 PyTorch 版 graph-based biaffine dependency parser 出发，整理训练流程，加入日志、checkpoint、实验历史记录和图表，并围绕词向量、优化器和编码器做了一组受控实验。

论文源码位于 `Paper/`。论文报告了在 8.3K 句中文依存句法树库上的十四组训练结果。主要发现是：在这个数据规模下，原始 BiLSTM 编码器仍然比从零训练的 Transformer 编码器更适合。最佳结果来自 BiLSTM + Muon + warmup + cosine decay，达到 86.93 dev UAS 和 68.37 dev LAS。我们训练的最大 SDPA 编码器达到 83.92 UAS，说明在小标注数据上，仅扩大模型容量并不能弥补数据不足。

## 我们做了什么

- 将项目整理成更可复现的训练流程，包含 checkpoint、JSON history、训练曲线和 run metadata。
- 对比了预训练词向量维度和词向量领域。
- 对比了 Adam、SGD 和 Muon 优化器。
- 增加了 SDPA Transformer 编码器变体，同时保持 biaffine scorer 和 MST decoder 不变。
- 诊断了两个重要训练问题：无 warmup 的 post-norm Transformer 会崩溃；Muon 默认学习率对小 Transformer 偏大。
- 将大体积训练产物单独发布到 Hugging Face，避免放进 GitHub。

## 仓库结构

```text
.
├── Paper/                                      # LaTeX 论文和图
├── PyTorch_Biaffine_Dependency_Parsing/        # 主代码
│   ├── main.py                                 # 训练入口
│   ├── trainer.py                              # 训练、评估、checkpoint 流程
│   ├── Config/config.cfg                       # 默认实验配置
│   ├── Dataloader/                             # CoNLL 数据读取和样本构造
│   ├── DataUtils/                              # 词表、batch、词向量、日志、checkpoint
│   ├── Model/Biaffine_Parsing/                 # 模型、biaffine 层、MST 解码
│   ├── scripts/                                # 服务器实验启动脚本
│   └── tests/                                  # 新增组件的单元测试
├── PROCESS.md                                  # 项目进度记录
├── PAPER.md                                    # 论文写作规划
├── requirements.txt                            # Python 依赖
└── LICENSE                                     # MIT 许可证
```

## 关键入口

训练入口：

```bash
cd PyTorch_Biaffine_Dependency_Parsing
python main.py --config ./Config/config.cfg --device cuda:0 --train -p
```

关键代码方向：

- `main.py` 解析命令行参数，加载数据，构建 parser 并启动训练。
- `trainer.py` 负责 epoch loop、优化器 step、评估、早停和 checkpoint 保存。
- `Model/Biaffine_Parsing/Model.py` 定义 encoder + MLP + biaffine scorer 的 forward。
- `Model/Biaffine_Parsing/Parser.py` 定义 loss 计算和解析接口。
- `Model/Biaffine_Parsing/MST.py` 将 arc scores 解码为合法依存树。
- `DataUtils/Logger.py` 和 `DataUtils/Checkpoint.py` 管理实验历史和模型产物。
- `scripts/launch_stage2_*.sh` 记录服务器上启动 Stage 2 编码器和优化器实验的命令。

计算量较大的实验运行在服务器 4：8 x A800 80GB GPU，CUDA 12.9，1TB RAM，数据盘挂载于 `/essfs100`。

## 训练产物

大模型文件不放在 GitHub。本项目完整 `Output/` 目录，包括 checkpoints、logs、plots 和 run summaries，发布在 Hugging Face：

[https://huggingface.co/kyksj/Dependency-Parsing](https://huggingface.co/kyksj/Dependency-Parsing)

代码仓库地址：

[https://github.com/kyksj-1/Dependency-Parsing-LXQ](https://github.com/kyksj-1/Dependency-Parsing-LXQ)

## 结果概览

| 配置 | Dev UAS | Dev LAS | 说明 |
|---|---:|---:|---|
| BiLSTM + Adam, 50 epochs | 86.43 | 67.65 | Stage-1 baseline |
| BiLSTM + Muon, 100 epochs | 86.93 | 68.37 | 最佳结果 |
| SDPA-9M + Adam, pre-norm + warmup | 82.00 | 63.55 | 修复后的小 Transformer |
| SDPA-Large + Muon | 83.92 | 65.60 | 最大编码器 |
| BiLSTM + SGD | 17.66 | 1.58 | 负控制，不收敛 |

核心结论很直接：在这个小规模监督树库上，BiLSTM 的归纳偏置非常有用。预训练词向量的覆盖价值大于不同词向量领域之间的细微差异。Muon 可以超过 Adam，但需要足够长的 schedule，让正交化更新完成后期细调。

## 接下来能做什么

- 用 `nn.LSTM` 替换当前 Python 循环版 `LSTMCell`，在保留近似 dropout 行为的同时提升训练速度。
- 增加 BERT-base-Chinese 或 MacBERT 编码器，测试预训练上下文表示能否缩小 Transformer 差距。
- 分别为循环网络和 Transformer 参数调 Muon 学习率。
- 补一个干净的 prediction CLI，用训练好的 checkpoint 解析原始句子或 CoNLL 文件。
- 对最好几组配置跑多 seed，估计方差。
- 在更大的标注树库或 Universal Dependencies Chinese 数据上复验。

## 许可证

本项目使用 MIT License。见 `LICENSE`。
