#!/bin/bash
# ============================================================================
# Stage 2 Phase 4 SOTA 大参数量训练 —— 服务器 4 单卡长时间运行
# ============================================================================
#
# 前置条件:
#   1. Phase 3 三个 run 已完成 (cuda:0/1/2 应当已空闲)
#   2. 词向量 sgns.mixed-large.300d.txt 仍在 Data/Embed/
#
# 用法 (服务器 4 上):
#   cd /essfs100/home/yangzhenjie/ljz/lxq/3句法分析实验代码
#   git fetch origin && git pull origin feat/stage2-sdpa-muon-20260511
#   bash PyTorch_Biaffine_Dependency_Parsing/scripts/launch_stage2_phase4.sh
#
# 模型配置 (TransformerEncoderLarge ~76M 参数):
#   - d_model=768, nhead=12, num_layers=8, ff_size=3072
#   - Pre-LN + RoPE + SwiGLU + LayerScale (init=1e-4)
#   - dropout=0.4 + embedding dropout=0.5
#
# 训练配置:
#   - 100 epoch + batch_size=32 + update_batch_size=4 (有效 batch=128)
#   - Muon optimizer + lr=0.02
#   - Warmup 1500 step + Cosine decay 到 1% base lr
#   - EMA decay=0.999 (评估用 EMA shadow 权重)
#
# 预计训练时间: A800 单卡 2-3 小时
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

# Phase 4 命令(单行)
SESSION="phase4_sota_large"
CMD="cd $ROOT && $CONDA_INIT && python -u main.py --device cuda:0 --tag enc_sdpa_large_sota --override Encoder.type=transformer_large Optimizer.adam=False Optimizer.muon=True Optimizer.sgd=False Optimizer.learning_rate=0.02 Train.epochs=100 Train.batch_size=32 Train.dev_batch_size=32 Train.test_batch_size=32 Train.update_batch_size=4 Train.log_interval=50 Train.use_warmup_cosine=True Train.warmup_steps=1500 Train.min_lr_ratio=0.01 Train.use_ema=True Train.ema_decay=0.999 Model.embed_dim=300 Model.dropout_emb=0.5 Embed.pretrained_embed_file=$WORDVEC 2>&1 | tee Output/_phase4_sota_large.log"

tmux kill-session -t "$SESSION" 2>/dev/null || true
tmux new-session -d -s "$SESSION" "$CMD; echo '=== DONE ==='; read"
echo "[OK] 启动 $SESSION (cuda:0)"
echo
echo "监控:"
echo "  tmux attach -t $SESSION"
echo "  tail -f $ROOT/Output/_phase4_sota_large.log"
echo "  watch -n 5 nvidia-smi"
