#!/bin/bash
# ============================================================================
# Stage 2 一键启动: Phase 3 三路 + Phase 4 SOTA 并行 (4 卡)
# ============================================================================
#
# 卡分配 (基于当前服务器空闲度,2026-05-11 实际观测):
#   cuda:4  Phase 3 Run 7: SDPA small + Adam
#   cuda:5  Phase 3 Run 8: BiLSTM    + Muon
#   cuda:7  Phase 3 Run 9: SDPA small + Muon
#   cuda:1  Phase 4 Run 10: SDPA large (~76M) + Muon + warmup + EMA
#
# 用法 (服务器 4 上):
#   cd /essfs100/home/yangzhenjie/ljz/lxq/3句法分析实验代码
#   git fetch origin && git checkout -f feat/stage2-sdpa-muon-20260511 && git pull
#   bash PyTorch_Biaffine_Dependency_Parsing/scripts/launch_stage2_now.sh
# ============================================================================

set -euo pipefail

ROOT="/essfs100/home/yangzhenjie/ljz/lxq/3句法分析实验代码/PyTorch_Biaffine_Dependency_Parsing"
EMB="./Data/Embed"
WORDVEC="${EMB}/sgns.mixed-large.300d.txt"

cd "$ROOT"

if [ ! -f "$WORDVEC" ]; then
    echo "[ERROR] 词向量不存在: $WORDVEC"
    exit 1
fi

CONDA_INIT='source /essfs100/home/yangzhenjie/miniconda3/etc/profile.d/conda.sh && conda activate ljz_env'

COMMON="Train.epochs=50 Train.batch_size=32 Train.dev_batch_size=32 Train.test_batch_size=32 Train.update_batch_size=4 Train.log_interval=50 Model.embed_dim=300 Embed.pretrained_embed_file=$WORDVEC"

# ============================================================================
# Phase 3 Run 7: SDPA small + Adam  on cuda:4
# ============================================================================
SESSION="phase3_sdpa_adam"
CMD="cd $ROOT && $CONDA_INIT && python -u main.py --device cuda:4 --tag enc_sdpa_adam --override Encoder.type=transformer Optimizer.adam=True Optimizer.muon=False Optimizer.sgd=False $COMMON 2>&1 | tee Output/_phase3_sdpa_adam.log"
tmux kill-session -t "$SESSION" 2>/dev/null || true
tmux new-session -d -s "$SESSION" "$CMD; echo '=== DONE ==='; sleep 86400"
echo "[OK] 启动 $SESSION (cuda:4)"

# ============================================================================
# Phase 3 Run 8: BiLSTM + Muon  on cuda:5
# ============================================================================
SESSION="phase3_lstm_muon"
CMD="cd $ROOT && $CONDA_INIT && python -u main.py --device cuda:5 --tag enc_lstm_muon --override Encoder.type=lstm Optimizer.adam=False Optimizer.muon=True Optimizer.sgd=False Optimizer.learning_rate=0.02 $COMMON 2>&1 | tee Output/_phase3_lstm_muon.log"
tmux kill-session -t "$SESSION" 2>/dev/null || true
tmux new-session -d -s "$SESSION" "$CMD; echo '=== DONE ==='; sleep 86400"
echo "[OK] 启动 $SESSION (cuda:5)"

# ============================================================================
# Phase 3 Run 9: SDPA small + Muon  on cuda:7
# ============================================================================
SESSION="phase3_sdpa_muon"
CMD="cd $ROOT && $CONDA_INIT && python -u main.py --device cuda:7 --tag enc_sdpa_muon --override Encoder.type=transformer Optimizer.adam=False Optimizer.muon=True Optimizer.sgd=False Optimizer.learning_rate=0.02 $COMMON 2>&1 | tee Output/_phase3_sdpa_muon.log"
tmux kill-session -t "$SESSION" 2>/dev/null || true
tmux new-session -d -s "$SESSION" "$CMD; echo '=== DONE ==='; sleep 86400"
echo "[OK] 启动 $SESSION (cuda:7)"

# ============================================================================
# Phase 4 Run 10: SDPA large (~76M) + Muon + warmup + EMA  on cuda:1
# ============================================================================
SESSION="phase4_sota_large"
CMD="cd $ROOT && $CONDA_INIT && python -u main.py --device cuda:1 --tag enc_sdpa_large_sota --override Encoder.type=transformer_large Optimizer.adam=False Optimizer.muon=True Optimizer.sgd=False Optimizer.learning_rate=0.02 Train.epochs=100 Train.batch_size=32 Train.dev_batch_size=32 Train.test_batch_size=32 Train.update_batch_size=4 Train.log_interval=50 Train.use_warmup_cosine=True Train.warmup_steps=1500 Train.min_lr_ratio=0.01 Train.use_ema=True Train.ema_decay=0.999 Model.embed_dim=300 Model.dropout_emb=0.5 Embed.pretrained_embed_file=$WORDVEC 2>&1 | tee Output/_phase4_sota_large.log"
tmux kill-session -t "$SESSION" 2>/dev/null || true
tmux new-session -d -s "$SESSION" "$CMD; echo '=== DONE ==='; sleep 86400"
echo "[OK] 启动 $SESSION (cuda:1)"

echo
echo "========================================================================"
echo "Stage 2 四路并行训练已启动。"
echo
echo "  tmux ls"
echo "  tmux attach -t phase3_sdpa_adam     # cuda:4, ~30-60 min"
echo "  tmux attach -t phase3_lstm_muon     # cuda:5, ~30-60 min"
echo "  tmux attach -t phase3_sdpa_muon     # cuda:7, ~30-60 min"
echo "  tmux attach -t phase4_sota_large    # cuda:1, ~2-3h"
echo "  Ctrl+B D 离开 tmux"
echo
echo "  tail -f $ROOT/Output/_phase4_sota_large.log    # 看 SOTA 进度"
echo "  nvidia-smi"
echo "========================================================================"
