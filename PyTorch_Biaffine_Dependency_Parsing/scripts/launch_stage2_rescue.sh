#!/bin/bash
# ============================================================================
# Stage 2 救场训练 v2 —— 修复 SDPA small 的 Post-LN 无 warmup 致命问题
# ============================================================================
#
# 背景:
#   2026-05-11 第一轮 Phase 3 SDPA small 跑出灾难性结果:
#     - enc_sdpa_adam: dev UAS 12.20 (train UAS 1-3% 纯随机,模型未学习)
#     - enc_sdpa_muon: dev UAS 53.35 (early stop, 后期退化)
#
# Root cause: Phase 3 small Transformer 用 Post-LN (norm_first=False) + 无 warmup
#   是 Vaswani 原版结构,众所周知必须有 warmup 否则训不动。我之前选 Post-LN 是
#   为了"和原版对齐",这是个错误判断。Phase 4 large 用 Pre-LN+warmup 所以训成
#   功 (UAS 83.92);small 也该用同样配置。
#
# 修复:
#   v2 救场加 norm_first=True (Pre-LN) + warmup 500 step + cosine decay
#   一切其它配置不变,保持公平对比。
#
# 用法:
#   bash PyTorch_Biaffine_Dependency_Parsing/scripts/launch_stage2_rescue.sh
# ============================================================================

set -euo pipefail

ROOT="/essfs100/home/yangzhenjie/ljz/lxq/3句法分析实验代码/PyTorch_Biaffine_Dependency_Parsing"
WORDVEC="./Data/Embed/sgns.mixed-large.300d.txt"

cd "$ROOT"

CONDA_INIT='source /essfs100/home/yangzhenjie/miniconda3/etc/profile.d/conda.sh && conda activate ljz_env'

# 清理旧 tmux session (已完成的 placeholder)
for s in phase3_sdpa_adam phase3_lstm_muon phase3_sdpa_muon phase4_sota_large; do
    tmux kill-session -t "$s" 2>/dev/null || true
done

COMMON_BASE="Train.epochs=50 Train.batch_size=32 Train.dev_batch_size=32 Train.test_batch_size=32 Train.update_batch_size=4 Train.log_interval=50 Model.embed_dim=300 Embed.pretrained_embed_file=$WORDVEC"
WARMUP="Encoder.norm_first=True Train.use_warmup_cosine=True Train.warmup_steps=500"

# ============================================================================
# R1: SDPA small + Pre-LN + Adam + warmup  on cuda:4
# ============================================================================
SESSION="rescue_sdpa_adam_v2"
CMD="cd $ROOT && $CONDA_INIT && python -u main.py --device cuda:4 --tag enc_sdpa_adam_v2 --override Encoder.type=transformer Optimizer.adam=True Optimizer.muon=False Optimizer.sgd=False $WARMUP Train.min_lr_ratio=0.1 $COMMON_BASE 2>&1 | tee Output/_rescue_sdpa_adam_v2.log"
tmux kill-session -t "$SESSION" 2>/dev/null || true
tmux new-session -d -s "$SESSION" "$CMD; echo '=== DONE ==='; sleep 86400"
echo "[OK] 启动 $SESSION (cuda:4)"

# ============================================================================
# R2: SDPA small + Pre-LN + Muon + cosine  on cuda:5
# ============================================================================
SESSION="rescue_sdpa_muon_v2"
CMD="cd $ROOT && $CONDA_INIT && python -u main.py --device cuda:5 --tag enc_sdpa_muon_v2 --override Encoder.type=transformer Optimizer.adam=False Optimizer.muon=True Optimizer.sgd=False Optimizer.learning_rate=0.02 $WARMUP Train.min_lr_ratio=0.05 $COMMON_BASE 2>&1 | tee Output/_rescue_sdpa_muon_v2.log"
tmux kill-session -t "$SESSION" 2>/dev/null || true
tmux new-session -d -s "$SESSION" "$CMD; echo '=== DONE ==='; sleep 86400"
echo "[OK] 启动 $SESSION (cuda:5)"

# ============================================================================
# R3: BiLSTM + Muon + warmup_cosine + 100 epoch (try beat 85.51)  on cuda:7
# ============================================================================
SESSION="rescue_lstm_muon_v2"
CMD="cd $ROOT && $CONDA_INIT && python -u main.py --device cuda:7 --tag enc_lstm_muon_v2 --override Encoder.type=lstm Optimizer.adam=False Optimizer.muon=True Optimizer.sgd=False Optimizer.learning_rate=0.02 Train.use_warmup_cosine=True Train.warmup_steps=500 Train.min_lr_ratio=0.05 Train.epochs=100 Train.batch_size=32 Train.dev_batch_size=32 Train.test_batch_size=32 Train.update_batch_size=4 Train.log_interval=50 Model.embed_dim=300 Embed.pretrained_embed_file=$WORDVEC 2>&1 | tee Output/_rescue_lstm_muon_v2.log"
tmux kill-session -t "$SESSION" 2>/dev/null || true
tmux new-session -d -s "$SESSION" "$CMD; echo '=== DONE ==='; sleep 86400"
echo "[OK] 启动 $SESSION (cuda:7)"

echo
echo "========================================================================"
echo "v2 救场训练已启动 3 路"
echo "  tmux ls"
echo "  tail -f $ROOT/Output/_rescue_sdpa_adam_v2.log"
echo "  tail -f $ROOT/Output/_rescue_sdpa_muon_v2.log"
echo "  tail -f $ROOT/Output/_rescue_lstm_muon_v2.log"
echo "========================================================================"
