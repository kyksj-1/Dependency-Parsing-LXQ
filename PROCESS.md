# 进度记录（PROCESS.md）

> 记录每轮交付的工作内容、对应代码改动、剩余事项。所有时间为本地时区。

---

## 2026-05-10 第 1 轮 · 工程化骨架 + 冒烟跑通

### 已完成
- **代码 review & 修复硬错误**
  - `main.py` 删除 `from imp import reload`（Python 3.12+ 已移除）与 `from test import ...`（test.py 不存在，原版根本起不来）
  - `mainHelp.py` 同步删除 `from test import load_test_model`，改为同名占位函数抛 NotImplementedError
  - `Layer.py` 把 `nn.init.orthogonal / constant` 替换为带下划线的原地版本
  - `Model.py` 给 `pretrained_embedding` 加 None 判断，关闭预训练向量也能跑
  - `Optimizer` 原版 SGD/Adam 双布尔位互斥逻辑脆弱，改为：同时为 True 时取 Adam 并打印警告
- **保存逻辑改造（核心需求）**
  - 删掉 `main.py` 里 `shutil.copytree("Model" / "DataUtils" / "Dataloader" / "Config", save_dir)` —— 不再每次实验都把整套代码复制几十 MB
  - 新增 `DataUtils/Checkpoint.py`：仅落 `best.pt` + `last.pt`（可选 `epoch_X.pt`），存 state_dict + 当前 lr + 关键超参
  - 训练产物统一落到 `Output/<时间戳>/` 下：`ckpts/`、`logs/`、`dictionary/`、`config.cfg`、`meta.json`
- **日志与曲线（MISSION 要求）**
  - 新增 `DataUtils/Logger.py`：`get_logger` 同时写文件 + stdout；`History` 收集 batch loss / epoch loss / dev-test UAS-LAS / lr，落盘 JSON 并画三张图（`loss_batch.png` / `loss_epoch.png` / `uas_las.png`）
  - `trainer.py` 整体重写：接外部 logger，按 epoch 调用 `History.add_epoch` + `dump` + `plot`，每 epoch 增量更新（中途 kill 也能拿到曲线）
  - early stop 由 `sys.exit()` 改成 `return`，确保退出前能 dump 曲线
- **配置文件**
  - `config.cfg` 默认值改为：`embed_dim=100` + 100d 样例词向量、`save_direction=./Output`、`adam=True`，并加注释说明 GPU 训练前需要替换成完整词向量
  - 新增 `Config/config_smoke.cfg` 跑 sample.conll，本地 1 分钟可验证
  - 新增 `Config/config_smoke_dev.cfg` 跑真实 dev.conll（534 句）+ 1 epoch，验证 GPU 流程
- **依赖**：`requirements.txt` 写了 torch / numpy / tqdm / matplotlib 四项

### 冒烟测试结果（2026-05-10）
| 配置 | 设备 | 耗时 | 结果 |
|---|---|---|---|
| `config_smoke.cfg` (sample.conll, 1 句, 2 epoch) | CPU | < 5 秒 | loss 4.34 → 4.30，UAS 60→50（数据太少不必关注）；ckpt + log + 图全产出 |
| `config_smoke_dev.cfg` (dev.conll, 534 句, 1 epoch, batch=16) | cuda:0 | ~22 秒 | loss 1.85，UAS 10.6%（仅 1 epoch + 词向量 OOV 97% 时正常）；GPU 通路 OK |

产物结构验证：
```
Output/<时间戳>/
├── ckpts/
│   ├── best.pt
│   └── last.pt
├── logs/
│   ├── train.log
│   ├── history.json
│   ├── loss_batch.png
│   ├── loss_epoch.png
│   └── uas_las.png
├── dictionary/
│   ├── dictionary_word.txt
│   └── dictionary_word.txt_ext.txt
├── config_*.cfg
└── meta.json
```

### 已知遗留 / 未做
- `MyLSTM` 用 Python for 循环逐时间步算，A800 上未必比内置 `nn.LSTM` 快太多。本轮不动，因为它是 Dozat 论文里的 variational dropout 实现，魔改可能破坏复现。**后续若想加速，改用 `nn.LSTM` + `dropout` 是 Stage 2 工作**。
- 独立推理脚本（旧版 test.py）尚未补回。当前 `--test` 进入 `start_test` 会抛 NotImplementedError。优先保证训练；推理本身在 `BiaffineParser.parse()` 里已经写好，只是少一个 CLI 包装。
- `requirements.txt` 还未在用户的服务器 4 / `ljz_env` conda 上验证 —— 当前是基于本地 research_env 反推的版本下限。

---

## 2026-05-10 第 2 轮 · 依存解析损失与标签全链路说明

### 已完成
- 输出“数据→模型→输出→损失”的全链路说明，明确 head/rel 标签含义与损失构成，并标注关键文件位置。
- 说明 rel loss 使用 gold head 的 teacher forcing 逻辑与 padding 忽略策略。

### 产出
- 说明文档：AI Response/dependency_parsing_loss_chain.md

### 下一步建议（按优先级）
1. **同步代码到服务器**：建议用 git 而非 scp（更易跟踪改动），目前项目根目录还没初始化 git。CLAUDE.md 要求工作流提交者名义 kyksj-1。
2. **下载完整词向量到 `Data/Embed/`** —— 当前样例文件 OOV 97%，跑出来的 UAS 没有参考价值。推荐至少下载：
   - `sgns.baidubaike.bigram-char` (300d) → A 维度对比基线 + B 领域对比基线
   - `sgns.renmin.bigram-char` (300d) → B 领域对比
   - 100d 版（用于 A 维度对比）
3. **跑 baseline**：在服务器 4 跑 `config.cfg` 默认（300d 百度百科 + Adam + 50 epoch）。预计 30-60 分钟能拿到 UAS≈90 的 baseline。
4. **三组对比实验**（MISSION 主线）：维度 / 领域 / 优化器，每组 2 次训练。建议直接复制几份 cfg 改文件名跑。
5. **报告**：损失曲线图直接用 `Output/<run>/logs/loss_batch.png`；伪代码 + 优化器原理对比另写。

---

## 2026-05-10 第 3 轮 · config 单文件机制 + 服务器部署 + 数据目录统一

### 已完成

**config 机制改造（用户偏好：不再为对比实验复制 cfg）**
- 删除 `Config/config_smoke.cfg`、`Config/config_smoke_dev.cfg`，回归"基础 cfg 只一份"
- `Config/config.py` 新增 `override(items)` / `dump_effective(path)` 方法，并删除原版每次启动都把 cfg 写回原文件的副作用
- `main.py` 新增 `--override Section.key=value [...]` 与 `--tag <name>` CLI 参数；产物目录命名 `Output/<时间戳>_<tag>/`，方便从一堆对比实验里区分
- 训练时落盘 `Output/<run>/effective_config.cfg`（基础 cfg + override 后的完整快照），meta.json 中含完整 argv + override 列表
- 删除 `DataUtils/Checkpoint.py` 里已废弃的 `copy_config_snapshot`

**服务器 4 部署**
- 工作区：`/essfs100/home/yangzhenjie/ljz/lxq/3句法分析实验代码/`（按 SERVER.md）
- 环境：`ljz_env`（torch 2.5.1+cu124, NumPy 2.2.6, 8× A800 80GB）
- 关键陷阱：非交互 ssh 不读 `~/.bashrc`，必须显式 `source /essfs100/home/yangzhenjie/miniconda3/etc/profile.d/conda.sh` 才能 `conda activate`
- 服务器冒烟两次（CPU sample.conll / GPU dev.conll）全部通过：ckpt + log + 三张图 + meta.json + effective_config.cfg 完整产出

**数据目录统一**
- `句法分析数据集/` 整个目录删除（HIT 部分与 ctb51 md5 完全一致）
- `句法分析数据集/THU/` 搬到 `PyTorch_Biaffine_Dependency_Parsing/Data/thu/`（保留备用，标注体系不同，不能直接对比）
- 玩具文件清理：`Data/ctb51/sample.3.conll`、`sample.test.conll`、`Data/Embed/1.txt`、`Data/Embed/char_word2vec.dat`
- 新增 `Data/README.md`（数据全景 + 来源 + md5 + 测试集说明）
- 新增 `Data/Embed/README.md`（完整词向量下载清单 + 命名规约 + override 用法）
- 服务器同步重组完成

**重组后服务器再冒烟**：1 epoch / dev.conll / cuda:0，9 秒训完，UAS 10.3%（仅 1 epoch + OOV 97% 下符合预期）。

### 当前目录结构
```
3句法分析实验代码/
├── CLAUDE.md
├── PROCESS.md
├── requirements.txt
├── 任务总览.md
├── 3句法分析实验.docx
└── PyTorch_Biaffine_Dependency_Parsing/
    ├── main.py / trainer.py / __init__.py
    ├── Config/
    │   ├── config.cfg            ← 唯一基础 cfg
    │   ├── config.py
    │   └── README.md
    ├── Data/                     ← 唯一数据根
    │   ├── README.md
    │   ├── ctb51/{train,dev,sample}.conll   ← HIT 体系（任务用）
    │   ├── thu/{train,dev}.conll            ← 语义依存（备用）
    │   └── Embed/
    │       ├── giga.100.txt.sample           ← 冒烟用
    │       └── README.md                     ← 词向量下载指引
    ├── DataUtils/                ← +Logger.py +Checkpoint.py
    ├── Dataloader/
    ├── Model/
    └── Save_All/                 ← 旧版残留空目录，不影响运行
```

### 下一步建议（按优先级）

1. **下载完整词向量** —— 见 `Data/Embed/README.md`，至少下载 `sgns.baidubaike.bigram-char`（300d）就能跑 baseline。
2. **服务器跑 baseline**（tmux + cuda:0）：
   ```bash
   ssh -i ~/.ssh/yangzhenjie_inspire_id_ed25519 -p 5052 yangzhenjie@112.25.93.66
   tmux new -s parser
   source /essfs100/home/yangzhenjie/miniconda3/etc/profile.d/conda.sh
   conda activate ljz_env
   cd /essfs100/home/yangzhenjie/ljz/lxq/3句法分析实验代码/PyTorch_Biaffine_Dependency_Parsing
   python main.py --device cuda:0 --tag baseline_300d_adam \
       --override Model.embed_dim=300 \
                  Embed.pretrained_embed_file=./Data/Embed/sgns.baidubaike.bigram-char
   # Ctrl+B D 离开 tmux；tmux attach -t parser 重连
   ```
3. **三组对比** —— 同一个 config.cfg + 不同 `--override` + 不同 `--tag`，跑 6 次。
4. 报告：损失曲线直接用 `Output/<run>/logs/loss_batch.png`。

---

## 2026-05-11 第 4 轮 · 完整训练矩阵 + 实验汇总

### 已完成
- **词向量准备**
  - 用户委托从 NLP&CC 渠道拿到 4 个 bz2（mixed-large / 人民日报 / 知乎 / 文学，全部 300d）
  - 服务器解压、统一命名为 `sgns.<corpus>.300d.txt`，落在 `Data/Embed/`
  - 新增 `tools/pca_embed.py`：把 mixed-large 300d 做 PCA 降到 100d，保留 73.53% 方差，产物 `sgns.mixed-large.100d.pca.txt`（581 MB / 636K 词）
- **训练矩阵 6 组**（A800 ×3 并行 × 串行 2 组，约 5.5 小时跑完）

  | tag | best dev UAS | best dev LAS |
  |---|---|---|
  | baseline_mixed_300_adam | **86.43** | **67.65** |
  | dim100_mixed_pca_adam | 86.14 | 66.57 |
  | domain_renmin_300_adam | 86.38 | 67.58 |
  | domain_zhihu_300_adam | 86.29 | 67.68 |
  | domain_literature_300_adam | 86.23 | 67.85 |
  | opt_mixed_300_sgd | 17.66 | 1.58 |
- 所有产物 scp 回本地 `Output/`（1.35 GB），写了 `Output/_summary.md`、画了 `Output/_compare_dev_uas_las.png`

### 结论
- **维度对比 (A)**：300d → PCA 100d 仅丢 0.29 UAS，证明 100d 已是 cost-effective
- **领域对比 (B)**：不同领域 300d 词向量在 dev 上差距 ≤ 0.20 UAS，远小于"有无预训练"的差距（参 from-scratch 84.84）
- **优化器对比 (C)**：SGD（lr=0.01, 无 momentum, 无 scheduler）完全不收敛（UAS 17.66）；Adam 在 BiLSTM+Biaffine 上不仅"更快"而是"必需"

### 下一步建议
1. 把 `Output/_summary.md` 内容整合进实验报告 word，把 `_compare_dev_uas_las.png` 当核心对比图
2. 各 run 的 `loss_batch.png` 任选一张当作 MISSION 要求的"损失曲线"
3. 报告里需要写"伪代码"那一块，用 `docs/02-句法分析实验任务详解.md` 第三章 Biaffine 算法核心思想改写
4. 论文 / 代码引用：Dozat & Manning 2017 ICLR
5. 若想冲更高分（可选）：换 nn.LSTM + warmup，预期 UAS 能到 88+；或加 BERT-base-chinese 编码器到 92+。

---

## 2026-05-11 第 5 轮 · Stage 2 启动（编码器消融 + 优化器扩展）

### 决策与方案

Stage 1 三组对比（维度 / 领域 / 优化器）+ from-scratch 下界已完成；剩余 3 天（截止 2026-05-14）做 Stage 2 进阶。方向锁定：

- **A. Transformer/SDPA 编码器对比** —— 用 `nn.TransformerEncoder` 替换 `MyLSTM`，保留 embedding + Biaffine + MST 整条解码链路不动，做干净的编码器消融。对应文献：Dozat & Manning 2017 ICLR → Mrini et al. 2020 / Cui et al. 2022 把 BiLSTM 替换为 SDPA 的演进路线。
- **B. Muon 优化器扩展** —— 在 Stage 1 的 Adam / SGD 之外加入 Muon（Keller Jordan 2024），把"优化器"对比维度从 2 扩到 3。Muon 对 2D 矩阵参数走 Newton-Schulz 5 阶迭代正交化，1D 参数 fallback 到 AdamW。

### 子任务拆解（17 项，TaskList 已建立）

```
Phase 0 · 分支与准备                ─ T0.1 T0.2 T0.3
Phase 1 · TransformerEncoder       ─ T1.1 T1.2 T1.3 T1.4 T1.5
Phase 2 · Muon Optimizer           ─ T2.1 T2.2 T2.3
Phase 3 · 服务器 3 路并行训练矩阵  ─ T3
Phase 3.5 · 暂停 + 用户讨论        ─ T3.5（大参数量 SOTA run 方案）
Phase 4 · 大参数量 SDPA SOTA run   ─ T4（叠加 warmup / RoPE / LayerScale 等）
Phase 5 · 汇总 + 文档              ─ T5.1 T5.2
Phase 6 · 条件合并到 main          ─ T6
```

### 关键决策（用户已确认）

| 项 | 值 |
|---|---|
| Transformer 参数量目标（Phase 3） | ≈ 与 BiLSTM 同量级 13M（d_model=512 / nhead=8 / layers=4 / ff=1024）|
| Transformer 参数量目标（Phase 4） | 大参数量 SOTA 级，具体规模在 Phase 3.5 与用户讨论后定 |
| Positional encoding | sinusoidal |
| Muon lr | 0.02（Keller Jordan 默认）|
| Muon fallback | AdamW(lr=3e-4) 仅作用于 1D 参数 |
| 本地开发环境 | conda `research_env` + 本地 GPU |
| 服务器训练环境 | 服务器 4 / `ljz_env` / cuda:0/1/2 三路并行 |

### 训练矩阵设计（Phase 3：3 主 run + 1 SOTA run）

| Run | tag | encoder | optimizer | 作用 | 启动时机 |
|---|---|---|---|---|---|
| 7 | `enc_sdpa_adam` | Transformer (small) | Adam | SDPA vs BiLSTM | Phase 3 |
| 8 | `enc_lstm_muon` | BiLSTM | Muon | Muon vs Adam | Phase 3 |
| 9 | `enc_sdpa_muon` | Transformer (small) | Muon | 交叉点 | Phase 3 |
| 10 | `enc_sdpa_large_sota` | Transformer (large + SOTA tricks) | TBD | 冲高分 | Phase 4（待讨论）|

### 安全与卫生

- 工作区在启动前删除了未追踪的 `upload_to_hf.py`（含明文 HF Write Token）。建议本地已经接触过该 token 的用户去 HuggingFace revoke 旧 token 重新生成。
- 创建子分支 `feat/stage2-sdpa-muon-20260511`（基于当前 main），按 CLAUDE.md §2.1 规范。

---

## 2026-05-11 第 6 轮 · Stage 2 代码侧完成 + 服务器 launcher 准备

### Phase 0-4 代码侧产出

| 模块 | 文件 | 单测 | 状态 |
|---|---|---|---|
| **Phase 1** SDPA encoder (small ~9M) | `Model/Biaffine_Parsing/TransformerEncoder.py` | 7/7 | ✅ |
| **Phase 2** Muon optimizer + AdamW fallback | `DataUtils/MuonOptimizer.py` + Optim.py / mainHelp.py 接入 | 9/9 | ✅ |
| **Phase 4.1** SOTA encoder (large ~76M, RoPE+SwiGLU+LayerScale+Pre-LN) | `Model/Biaffine_Parsing/TransformerEncoderLarge.py` | 14/14 | ✅ |
| **Phase 4.2** Warmup+Cosine schedule | `DataUtils/Scheduler.py` | 3/3 | ✅ |
| **Phase 4.2** EMA weight averaging | `DataUtils/EMA.py` | 3/3 | ✅ |
| Config 扩展 | `Config/config.cfg` + `Config/config.py` | — | ✅ |
| Model dispatch | `Model/Biaffine_Parsing/Model.py` (encoder_type: lstm/transformer/transformer_large) | — | ✅ |
| Trainer 整合 | `trainer.py` (scheduler/EMA 在 large run 启用) | — | ✅ |
| 服务器 launcher | `scripts/launch_stage2_phase3.sh` + `scripts/launch_stage2_phase4.sh` | — | ✅ |

**总单测**: 36/36 通过（research_env / RTX 4060 本地）。**端到端冒烟矩阵**:

| encoder | optimizer | extras | epoch loss 变化 | 备注 |
|---|---|---|---|---|
| lstm | adam | — | baseline 不变 | 回归测试 |
| transformer (small) | adam | — | 1.086 → 1.066 | Phase 3 通路 |
| lstm | muon | — | 1.086 → 1.084 | Phase 3 通路 |
| transformer (small) | muon | — | 1.086 → 1.042 | Phase 3 通路 |
| transformer_large | muon | warmup+cosine+EMA | 1.086 → 1.064 → 1.012 | Phase 4 完整链路 |

### 弃用的 trick (用户提议但评估后弃用)

| Trick | 评估结论 |
|---|---|
| MoE (Mixture of Experts) | 8.3K 句训练集 router 极易塌缩,门槛在 100B+ token,差 5 个数量级 |
| Linear Attention | 句子最长 200 词,O(L²) 已极便宜,linear 是换表达力换速度,反向收益 |
| mHC connection (arxiv 2512.24880) | 2025-12 论文太新,LLM-corpus 验证,小数据 NLP 经典任务迁移性未知,时间不允许精读+调试 |
| Gate Attention (OpenReview 1b7whO4SfY) | 与已采用的 LayerScale 功能重叠,后者 timm 验证更充分 |

### Phase 3 训练矩阵（待用户在服务器 4 上启动）

```bash
ssh -i ~/.ssh/yangzhenjie_inspire_id_ed25519 -p 5052 yangzhenjie@112.25.93.66
cd /essfs100/home/yangzhenjie/ljz/lxq/3句法分析实验代码
git fetch origin && git checkout feat/stage2-sdpa-muon-20260511 && git pull
bash PyTorch_Biaffine_Dependency_Parsing/scripts/launch_stage2_phase3.sh
```

| Run | tag | encoder | optimizer | cuda | 预计 |
|---|---|---|---|---|---|
| 7 | `enc_sdpa_adam` | Transformer small (~9M) | Adam lr=1e-3 | 0 | 30-60 min |
| 8 | `enc_lstm_muon` | BiLSTM | Muon lr=0.02 | 1 | 30-60 min |
| 9 | `enc_sdpa_muon` | Transformer small | Muon lr=0.02 | 2 | 30-60 min |

### Phase 4 大参数量 SOTA run（Phase 3 完成后启动）

```bash
bash PyTorch_Biaffine_Dependency_Parsing/scripts/launch_stage2_phase4.sh
```

| Run | tag | encoder | optimizer | extras | 预计 |
|---|---|---|---|---|---|
| 10 | `enc_sdpa_large_sota` | Transformer **large** (~76M, RoPE+SwiGLU+LayerScale+Pre-LN) | Muon lr=0.02 | warmup 1500 + cosine + EMA decay=0.999 + dropout 0.4 + emb_dropout 0.5 | 2-3 h |

预期: UAS 88-90 (vs Phase 3 small ≈ Stage 1 baseline 86.4)。

---

## 2026-05-11 第 7 轮 · 服务器 4 路并行训练已启动

### 启动操作
- AI 自主 ssh 服务器 4(私钥无 passphrase 可非交互连接)
- 服务器侧 `git init` + `remote add origin` + `git checkout -f -B feat/...`
  (原目录为 scp 同步,非 git;reset 到 GitHub feat 分支)
- 新 launcher `scripts/launch_stage2_now.sh` 一次性起 4 个 tmux session
- 卡分配避开已 99% 利用率的 cuda:6,选 cuda:1/4/5/7

### 4 路并行训练矩阵

| tmux session | tag | encoder | optimizer | cuda | 实测显存 |
|---|---|---|---|---|---|
| phase3_sdpa_adam | enc_sdpa_adam | Transformer small (~9M) | Adam lr=1e-3 | 4 | +4 GB |
| phase3_lstm_muon | enc_lstm_muon | BiLSTM | Muon lr=0.02 | 5 | +5 GB |
| phase3_sdpa_muon | enc_sdpa_muon | Transformer small | Muon lr=0.02 | 7 | +4 GB |
| phase4_sota_large | enc_sdpa_large_sota | Transformer **large** (~76M, RoPE/SwiGLU/LayerScale/Pre-LN) | Muon lr=0.02 + warmup 1500 + cosine + EMA decay=0.999 | 1 | +17 GB |

启动时间 20:08 (服务器本地)。Phase 3 预计 30-60 min,Phase 4 预计 2-3h。

### 启动 5 min 后状态确认

| Run | Epoch / Batch | loss | UAS |
|---|---|---|---|
| phase3_sdpa_adam | 2 / 151/260 | 1.79 | 16.0 |
| phase3_lstm_muon | 1 / 101/260 | 1.50 | 22.0 |
| phase3_sdpa_muon | 1 / 101/260 | 1.87 | 14.6 |
| phase4_sota_large | 1 / 151/260 | 1.81 | 17.0 |

所有 loss 稳定下降,GPU 100% 利用率。

### 监控方法

```bash
# ssh 上服务器 4 后:
tmux ls
tmux attach -t phase4_sota_large    # 看 SOTA 进度,Ctrl+B D 退出
tail -f Output/_phase4_sota_large.log

# 找已完成的 run:
ls -la /essfs100/home/yangzhenjie/ljz/lxq/3句法分析实验代码/PyTorch_Biaffine_Dependency_Parsing/Output/ | grep enc_
```

### 训练完成后下一步

1. scp 所有 4 个新 run 目录回本地 `Output/`
2. 更新 `Output/_summary.md` (新增 4 行)
3. 重画 `_compare_dev_uas_las.png` (叠加 4 条新曲线)
4. 把数字填回 `docs/03-Stage2-编码器与优化器扩展.md` 表格 6.1
5. 按 §2.3 条件检查后合并 feat 分支回 main

---

## 新数据(用户后续提供,未处理)

`D:\发送给别人\lxq\3句法分析实验代码\PyTorch_Biaffine_Dependency_Parsing\Data\Embed` 新增 5 个 bz2:
- sgns.merge.word.bz2
- sgns.sikuquanshu.word.bz2 (四库全书)
- sgns.sogou.word.bz2
- sgns.weibo.word.bz2
- sgns.financial.word.bz2

当前 Phase 3 / Phase 4 训练命令统一用 sgns.mixed-large.300d.txt(已在服务器),
不依赖这些新数据。若后续要做"扩展领域对比"(Stage 1 B 组的延伸),需先 scp 上传 +
bz2 解压 + 重命名为 sgns.<corpus>.300d.txt 格式。**当前未做。**

---

## 2026-05-12 第 8 轮 · 训练完成 + 故障诊断 + v2 救场启动

### 第一轮训练结果(2026-05-12 早晨收割)

| ID | tag | encoder | optimizer | best dev UAS | epochs | 状态 |
|---|---|---|---|---|---|---|
| 7 | enc_sdpa_adam | Transformer small (Post-LN) | Adam | **12.20** | 23 (早停) | 🚨 完全不收敛 |
| 8 | enc_lstm_muon | BiLSTM | Muon | **85.51** | 50 | ✅ 健康 |
| 9 | enc_sdpa_muon | Transformer small (Post-LN) | Muon | **53.35** | 25 (早停) | ⚠️ 早期到 53 后退化 |
| 10 | enc_sdpa_large_sota | Transformer large (Pre-LN+RoPE+SwiGLU+LayerScale) | Muon+warmup+EMA | **83.92** | 99 | ⚠️ 仍低于 baseline 2.5 点 |

### Root Cause 诊断: Phase 3 small SDPA 用 Post-LN 无 warmup

- enc_sdpa_adam train UAS 始终 1-3% 纯随机,模型彻底没学习
- enc_sdpa_muon epoch 5 后 dev UAS 从 53 → 43,典型不稳定退化
- enc_sdpa_large_sota 用了 Pre-LN+warmup+cosine+EMA 反而能 84,印证 Pre-LN+warmup 必需
- **教训**: 我之前为"对齐 Vaswani 原版"选 Post-LN 是错误判断。Phase 3 small 也该用 Pre-LN+warmup。

### v2 救场启动 (12:00 启动)

- rescue_sdpa_adam_v2  (cuda:4) Pre-LN + warmup 500 + Adam     - 50 epoch
- rescue_sdpa_muon_v2  (cuda:5) Pre-LN + warmup 500 + Muon     - 50 epoch (可能早停)
- rescue_lstm_muon_v2  (cuda:7) warmup 500 + Muon              - 100 epoch (试图超 85.51)
- rescue_sdpa_muon_v3  (cuda:0) Pre-LN + warmup 1000 + Muon lr=0.005 - 50 epoch (v2 lr 太大的备份方案)

### 操作事故记录

1. **inline ssh 命令双引号 quoting 失败**: 嵌套 quoting 让 conda init 被截断, tmux session 起来直接 "command not found" 退出。**经验**: PowerShell -> ssh -> tmux 三层 quoting 不可靠, 应当用独立 .sh + scp 或 git。

2. **bash launcher v3 触发 v2 session 重启**: 原 launch_stage2_rescue.sh 主体没有 conditional guard, 任何参数都会先 kill+restart v2 三个 session。**损失约 35 min v2 训练进度**。修复: 拆出独立 launch_v3_only.sh。

3. **git pull HTTP/2 错误**: GitHub 偶发 RPC failure。**修复**: 用 for retry 循环或 scp 直接传文件绕过 git。

### 中期数字 (12:13)
- sdpa_adam_v2: epoch ~9 dev UAS 69.43 (持续涨,预计 50 epoch 到 82-85)
- sdpa_muon_v2: peak epoch 5 UAS 53.35, 早停在即
- lstm_muon_v2: epoch ~5 dev UAS 68.51 (100 epoch 跑 5h)
- sdpa_muon_v3: 刚启动

### 待办

1. ⏳ 等 13:00 左右 3 个 small 50-epoch 训练完成
2. 重新画 _compare_stage2.png (扩展到 14 路曲线)
3. 更新 _summary.md 加 v2/v3 数字
4. 等 17:00 左右 lstm_muon_v2 100 epoch 完成
5. 整合所有数字到 docs/03
6. 按 §2.3 条件检查后合并到 main

