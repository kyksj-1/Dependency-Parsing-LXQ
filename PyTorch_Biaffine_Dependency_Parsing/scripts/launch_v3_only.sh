#!/bin/bash
# 独立 v3 启动器,不动 v2 session
set -euo pipefail

ROOT="/essfs100/home/yangzhenjie/ljz/lxq/3句法分析实验代码/PyTorch_Biaffine_Dependency_Parsing"
cd "$ROOT"

SESSION="rescue_sdpa_muon_v3"
CONDA_INIT='source /essfs100/home/yangzhenjie/miniconda3/etc/profile.d/conda.sh && conda activate ljz_env'

CMD="cd $ROOT && $CONDA_INIT && python -u main.py --device cuda:0 --tag enc_sdpa_muon_v3_lr005 --override Encoder.type=transformer Encoder.norm_first=True Optimizer.adam=False Optimizer.muon=True Optimizer.sgd=False Optimizer.learning_rate=0.005 Train.use_warmup_cosine=True Train.warmup_steps=1000 Train.min_lr_ratio=0.05 Train.epochs=50 Train.batch_size=32 Train.dev_batch_size=32 Train.test_batch_size=32 Train.update_batch_size=4 Train.log_interval=50 Model.embed_dim=300 Embed.pretrained_embed_file=./Data/Embed/sgns.mixed-large.300d.txt 2>&1 | tee Output/_rescue_sdpa_muon_v3.log"

tmux kill-session -t "$SESSION" 2>/dev/null || true
tmux new-session -d -s "$SESSION" "$CMD; echo '=== DONE ==='; sleep 86400"
echo "[OK] 启动 $SESSION (cuda:0, lr=0.005, warmup 1000)"
tmux ls | grep rescue
