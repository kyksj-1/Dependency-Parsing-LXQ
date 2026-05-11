#!/bin/bash
# ============================================================================
# Stage 2 Phase 3 训练矩阵 —— 服务器 4 (A800 ×8) 三路并行启动脚本
# ============================================================================
#
# 用法 (服务器 4 上,任意位置)：
#   ssh -i ~/.ssh/yangzhenjie_inspire_id_ed25519 -p 5052 yangzhenjie@112.25.93.66
#   cd /essfs100/home/yangzhenjie/ljz/lxq/3句法分析实验代码
#   git fetch origin && git checkout feat/stage2-sdpa-muon-20260511 && git pull
#   bash PyTorch_Biaffine_Dependency_Parsing/scripts/launch_stage2_phase3.sh
#
# 脚本动作:
#   1. 起 3 个独立 tmux 会话:phase3_sdpa_adam / phase3_lstm_muon / phase3_sdpa_muon
#   2. 每个会话在对应 cuda 卡上跑 50 epoch + batch_size=32
#   3. 退出 tmux 用 Ctrl+B D;重连 `tmux attach -t <name>`
#
# 跑完产物在 PyTorch_Biaffine_Dependency_Parsing/Output/<时间戳>_<tag>/
# ============================================================================

set -euo pipefail

# ---------- 路径与环境 ----------
ROOT="/essfs100/home/yangzhenjie/ljz/lxq/3句法分析实验代码/PyTorch_Biaffine_Dependency_Parsing"
EMB="./Data/Embed"
WORDVEC="${EMB}/sgns.mixed-large.300d.txt"

cd "$ROOT"

# 校验环境
if [ ! -f "$WORDVEC" ]; then
    echo "[ERROR] 词向量不存在: $WORDVEC"
    echo "请确认 Data/Embed/sgns.mixed-large.300d.txt 已就位 (Stage 1 已下好)"
    exit 1
fi

# Conda 激活引导 (非交互 ssh 不会自动读 ~/.bashrc)
CONDA_INIT='source /essfs100/home/yangzhenjie/miniconda3/etc/profile.d/conda.sh && conda activate ljz_env'

# 公共训练参数
COMMON='Train.epochs=50 Train.batch_size=32 Train.dev_batch_size=32 Train.test_batch_size=32 Train.update_batch_size=4 Train.log_interval=50 Model.embed_dim=300 Embed.pretrained_embed_file='"$WORDVEC"

# ============================================================================
# Run 7: Transformer (small ~9M) + Adam      [cuda:0]
# ============================================================================
SESSION="phase3_sdpa_adam"
CMD="cd $ROOT && $CONDA_INIT && python -u main.py --device cuda:0 --tag enc_sdpa_adam --override Encoder.type=transformer Optimizer.adam=True Optimizer.muon=False Optimizer.sgd=False $COMMON 2>&1 | tee Output/_phase3_sdpa_adam.log"
tmux kill-session -t "$SESSION" 2>/dev/null || true
tmux new-session -d -s "$SESSION" "$CMD; echo '=== DONE, press any key ==='; read"
echo "[OK] 启动 $SESSION (cuda:0)"

# ============================================================================
# Run 8: BiLSTM + Muon                       [cuda:1]
# ============================================================================
SESSION="phase3_lstm_muon"
CMD="cd $ROOT && $CONDA_INIT && python -u main.py --device cuda:1 --tag enc_lstm_muon --override Encoder.type=lstm Optimizer.adam=False Optimizer.muon=True Optimizer.sgd=False Optimizer.learning_rate=0.02 $COMMON 2>&1 | tee Output/_phase3_lstm_muon.log"
tmux kill-session -t "$SESSION" 2>/dev/null || true
tmux new-session -d -s "$SESSION" "$CMD; echo '=== DONE, press any key ==='; read"
echo "[OK] 启动 $SESSION (cuda:1)"

# ============================================================================
# Run 9: Transformer (small ~9M) + Muon      [cuda:2]
# ============================================================================
SESSION="phase3_sdpa_muon"
CMD="cd $ROOT && $CONDA_INIT && python -u main.py --device cuda:2 --tag enc_sdpa_muon --override Encoder.type=transformer Optimizer.adam=False Optimizer.muon=True Optimizer.sgd=False Optimizer.learning_rate=0.02 $COMMON 2>&1 | tee Output/_phase3_sdpa_muon.log"
tmux kill-session -t "$SESSION" 2>/dev/null || true
tmux new-session -d -s "$SESSION" "$CMD; echo '=== DONE, press any key ==='; read"
echo "[OK] 启动 $SESSION (cuda:2)"

echo
echo "========================================================================"
echo "Phase 3 三路训练全部已启动。监控方式:"
echo "  tmux ls                              # 列出所有 session"
echo "  tmux attach -t phase3_sdpa_adam      # 进 Run 7"
echo "  tmux attach -t phase3_lstm_muon      # 进 Run 8"
echo "  tmux attach -t phase3_sdpa_muon      # 进 Run 9"
echo "  Ctrl+B D                              # 离开 tmux (任务继续)"
echo
echo "  tail -f $ROOT/Output/_phase3_sdpa_adam.log    # 不进 tmux 直接看日志"
echo
echo "  watch -n 5 nvidia-smi                # 看 GPU 利用率"
echo "========================================================================"
