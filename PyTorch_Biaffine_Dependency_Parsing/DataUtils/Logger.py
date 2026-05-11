# -*- coding: utf-8 -*-
"""
文件说明：训练日志与历史记录工具
功能：
    - get_logger: 同时把日志写到文件和控制台，避免每次实验丢失训练曲线
    - History: 记录 batch 级 loss、epoch 级 train/dev/test 指标，支持落盘 JSON 与画图
之所以单独抽出来，是因为原代码只把 loss 打到 stdout，跑完就丢，
MISSION 又明确要求"绘制损失曲线"——单点改造避免在 trainer.py 里堆细节。
"""

import json
import logging
import os
from typing import List, Optional


def get_logger(log_path: str, name: str = "biaffine") -> logging.Logger:
    """
    返回一个同时写文件 + 控制台的 logger
    :param log_path: 日志文件路径（自动创建父目录）
    :param name: logger 名，避免重复 addHandler
    """
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    # 防止重复添加 handler（多次 import / 重新 init 时）
    logger.handlers.clear()

    fmt = logging.Formatter("[%(asctime)s] %(levelname)s %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")

    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    logger.propagate = False
    return logger


class History:
    """
    训练曲线收集器：把 batch / epoch 级指标存起来，最后落盘 + 画图。
    用最朴素的 list，不引入 tensorboard / wandb，避免额外依赖。
    """

    def __init__(self, save_dir: str):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        self.batch_losses: List[float] = []   # 每个 batch 的 loss（已按 update_batch_size 缩放）
        self.epoch_losses: List[float] = []   # 每个 epoch 的平均 loss
        self.train_uas: List[float] = []
        self.train_las: List[float] = []
        self.dev_uas: List[float] = []
        self.dev_las: List[float] = []
        self.test_uas: List[float] = []
        self.test_las: List[float] = []
        self.lr: List[float] = []

    def add_batch_loss(self, loss: float) -> None:
        self.batch_losses.append(float(loss))

    def add_epoch(self, epoch_loss: float, train_uas: float, train_las: float,
                  dev_uas: float, dev_las: float,
                  test_uas: Optional[float], test_las: Optional[float],
                  lr: float) -> None:
        self.epoch_losses.append(float(epoch_loss))
        self.train_uas.append(float(train_uas))
        self.train_las.append(float(train_las))
        self.dev_uas.append(float(dev_uas))
        self.dev_las.append(float(dev_las))
        self.test_uas.append(float(test_uas) if test_uas is not None else float("nan"))
        self.test_las.append(float(test_las) if test_las is not None else float("nan"))
        self.lr.append(float(lr))

    def dump(self) -> str:
        """把所有 history 序列化为 JSON 落盘，返回文件路径"""
        path = os.path.join(self.save_dir, "history.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump({
                "batch_losses": self.batch_losses,
                "epoch_losses": self.epoch_losses,
                "train_uas": self.train_uas,
                "train_las": self.train_las,
                "dev_uas": self.dev_uas,
                "dev_las": self.dev_las,
                "test_uas": self.test_uas,
                "test_las": self.test_las,
                "lr": self.lr,
            }, f, ensure_ascii=False, indent=2)
        return path

    def plot(self) -> List[str]:
        """画 loss 曲线 + UAS/LAS 曲线。matplotlib 在服务器上也能用 Agg 后端。"""
        try:
            import matplotlib
            matplotlib.use("Agg")  # 服务器没显示器时必须 Agg
            import matplotlib.pyplot as plt
        except ImportError:
            return []

        out_paths: List[str] = []

        # 1) batch 级 loss（最细粒度，画出来就是 MISSION 要求的损失曲线）
        if self.batch_losses:
            plt.figure(figsize=(8, 4))
            plt.plot(self.batch_losses, linewidth=0.6)
            plt.xlabel("batch step")
            plt.ylabel("loss")
            plt.title("Training Loss (per batch)")
            plt.tight_layout()
            p = os.path.join(self.save_dir, "loss_batch.png")
            plt.savefig(p, dpi=150)
            plt.close()
            out_paths.append(p)

        # 2) epoch 级 loss
        if self.epoch_losses:
            plt.figure(figsize=(8, 4))
            xs = list(range(1, len(self.epoch_losses) + 1))
            plt.plot(xs, self.epoch_losses, marker="o")
            plt.xlabel("epoch")
            plt.ylabel("avg loss")
            plt.title("Training Loss (per epoch)")
            plt.tight_layout()
            p = os.path.join(self.save_dir, "loss_epoch.png")
            plt.savefig(p, dpi=150)
            plt.close()
            out_paths.append(p)

        # 3) UAS/LAS 曲线
        if self.dev_uas:
            plt.figure(figsize=(8, 4))
            xs = list(range(1, len(self.dev_uas) + 1))
            plt.plot(xs, self.dev_uas, marker="o", label="dev UAS")
            plt.plot(xs, self.dev_las, marker="s", label="dev LAS")
            if self.test_uas:
                plt.plot(xs, self.test_uas, marker="^", label="test UAS", linestyle="--")
                plt.plot(xs, self.test_las, marker="v", label="test LAS", linestyle="--")
            plt.xlabel("epoch")
            plt.ylabel("score (%)")
            plt.title("Eval UAS / LAS")
            plt.legend()
            plt.tight_layout()
            p = os.path.join(self.save_dir, "uas_las.png")
            plt.savefig(p, dpi=150)
            plt.close()
            out_paths.append(p)

        return out_paths
