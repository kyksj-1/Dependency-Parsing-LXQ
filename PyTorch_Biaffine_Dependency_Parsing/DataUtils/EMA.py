"""
Exponential Moving Average (EMA) of model parameters.

Polyak averaging (Polyak & Juditsky 1992),在 inference 时用 EMA 权重通常能稳定提升 0.2-0.5 个点。

用法:
    ema = EMAWrapper(model, decay=0.999)
    # 训练循环:
    for batch in ...:
        loss.backward()
        optimizer.step()
        ema.update()          # 每个 optimizer.step() 后调一次

    # 评估时:
    with ema.swap():          # 暂时把 EMA 权重 swap 进 model
        eval(model)           # 用 EMA 权重做评估
    # context 退出后自动恢复原 model 权重
"""

from contextlib import contextmanager
from typing import Dict

import torch
import torch.nn as nn


class EMAWrapper:
    """
    维护 model 所有可训练参数的指数移动平均。

    内存开销: 大约等于 model 参数量的 1 倍 (shadow 权重)。
    对 70M model 来说 EMA shadow 约 280 MB (fp32),A800 80GB 完全不在话下。
    """

    def __init__(self, model: nn.Module, decay: float = 0.999):
        """
        Args:
            model: 被跟踪的模型
            decay: EMA 衰减系数,越大越平滑 (典型 0.999 / 0.9999)
                   公式: shadow <- decay * shadow + (1-decay) * model_param
        """
        self.model = model
        self.decay = decay
        # shadow 参数 (拷贝一份初始权重作为起点)
        self.shadow: Dict[str, torch.Tensor] = {}
        for name, p in model.named_parameters():
            if p.requires_grad:
                self.shadow[name] = p.detach().clone()
        # backup 仅在 swap 时临时用
        self.backup: Dict[str, torch.Tensor] = {}

    @torch.no_grad()
    def update(self) -> None:
        """每次 optimizer.step() 之后调用一次,把 model 当前权重融进 shadow"""
        for name, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            shadow = self.shadow[name]
            # in-place: shadow <- decay * shadow + (1-decay) * p
            shadow.mul_(self.decay).add_(p.detach(), alpha=1.0 - self.decay)

    @contextmanager
    def swap(self):
        """
        Context manager: 暂时把 EMA 权重 swap 进 model,退出时恢复。
        典型用法:
            with ema.swap():
                eval_results = evaluate(model)
        """
        self._apply_shadow()
        try:
            yield
        finally:
            self._restore()

    def _apply_shadow(self) -> None:
        for name, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            self.backup[name] = p.detach().clone()
            p.data.copy_(self.shadow[name])

    def _restore(self) -> None:
        for name, p in self.model.named_parameters():
            if name in self.backup:
                p.data.copy_(self.backup[name])
        self.backup.clear()

    def state_dict(self) -> dict:
        return {
            "decay": self.decay,
            "shadow": {k: v.clone() for k, v in self.shadow.items()},
        }

    def load_state_dict(self, sd: dict) -> None:
        self.decay = sd["decay"]
        self.shadow = {k: v.clone() for k, v in sd["shadow"].items()}
