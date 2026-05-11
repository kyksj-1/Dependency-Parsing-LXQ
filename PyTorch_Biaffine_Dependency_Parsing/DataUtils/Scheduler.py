"""
Warmup + Cosine Decay learning rate scheduler.

仅在 Phase 4 SOTA run 启用,通过 config.use_warmup_cosine = True 开关。

设计:
    - 前 warmup_steps 步: lr 线性从 0 上升到 init_lr
    - 之后到 total_steps: lr 按 cosine 从 init_lr 衰减到 min_lr
    - step() 在每次 optimizer.step() 之后调用,直接改 param_groups[i]['lr']

支持 Muon 的混合 param_groups (Muon group 和 AdamW group 各自有不同 base lr):
    每个 group 按其 base_lr 独立按比例缩放,scheduler 不破坏 base_lr 的相对关系
"""

import math
from typing import List


class WarmupCosineSchedule:
    """
    LR scheduler with linear warmup + cosine decay.

    用法:
        sched = WarmupCosineSchedule(optimizer, warmup_steps=1500, total_steps=80000, min_lr_ratio=0.01)
        for batch in train_iter:
            optimizer.step()
            sched.step()
    """

    def __init__(
        self,
        optimizer,
        warmup_steps: int,
        total_steps: int,
        min_lr_ratio: float = 0.01,
    ):
        """
        Args:
            optimizer:     被调度的 optimizer (支持 PyTorch 接口的任意对象,含 MuonWithAdamW)
            warmup_steps:  前多少 step 做线性 warmup
            total_steps:   总训练 step 数
            min_lr_ratio:  cosine decay 的下限,base_lr * min_lr_ratio
        """
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = max(total_steps, warmup_steps + 1)
        self.min_lr_ratio = min_lr_ratio
        self.current_step = 0

        # 记录每个 param_group 的 base_lr,作为后续缩放基准
        self.base_lrs: List[float] = [
            float(g["lr"]) for g in optimizer.param_groups
        ]

    def _lr_scale(self, step: int) -> float:
        """返回当前 step 的 lr 缩放比例 (相对于 base_lr)"""
        if step < self.warmup_steps:
            # 线性 warmup: 0 -> 1
            return step / max(self.warmup_steps, 1)
        # cosine decay: 1 -> min_lr_ratio
        progress = (step - self.warmup_steps) / max(self.total_steps - self.warmup_steps, 1)
        progress = min(progress, 1.0)
        cos_val = 0.5 * (1 + math.cos(math.pi * progress))
        # 从 1 衰减到 min_lr_ratio
        return self.min_lr_ratio + (1 - self.min_lr_ratio) * cos_val

    def step(self) -> float:
        """前进一步,更新 optimizer.param_groups 中所有 group 的 lr。返回当前 scale"""
        self.current_step += 1
        scale = self._lr_scale(self.current_step)
        for group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            group["lr"] = base_lr * scale
        return scale

    def get_last_lr(self) -> List[float]:
        return [g["lr"] for g in self.optimizer.param_groups]

    def state_dict(self) -> dict:
        return {
            "current_step": self.current_step,
            "warmup_steps": self.warmup_steps,
            "total_steps": self.total_steps,
            "min_lr_ratio": self.min_lr_ratio,
            "base_lrs": self.base_lrs,
        }

    def load_state_dict(self, sd: dict) -> None:
        self.current_step = sd["current_step"]
        self.warmup_steps = sd["warmup_steps"]
        self.total_steps = sd["total_steps"]
        self.min_lr_ratio = sd["min_lr_ratio"]
        self.base_lrs = sd["base_lrs"]
