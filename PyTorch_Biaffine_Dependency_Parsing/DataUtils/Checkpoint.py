# -*- coding: utf-8 -*-
"""
文件说明：模型 checkpoint 工具
功能：
    - save_checkpoint：把 (model_state, optimizer_state, epoch, best_score, config_snapshot) 一起存
    - load_checkpoint：从 .pt 文件恢复训练或推理用
设计目标：取代旧版 `Save_All/Save_model/<时间戳>/` 复制整套代码 + 多个 .pt 的笨重做法，
        统一只存 best.pt / last.pt 两个文件，体积可控、易迁移。
"""

import os
import shutil
import time
from typing import Any, Dict, Optional

import torch
import torch.nn as nn


def save_checkpoint(model: nn.Module,
                    save_path: str,
                    epoch: int,
                    best_score: Optional[float] = None,
                    extra: Optional[Dict[str, Any]] = None) -> str:
    """
    保存一个 ckpt。
    :param model: 已训练的模型（保存 state_dict 即可，不存整个对象避免 pickle 类路径锁定）
    :param save_path: 完整文件路径，调用方决定是 best.pt 还是 epoch_X.pt
    :param epoch: 当前 epoch
    :param best_score: 关联的最佳 dev 分数，便于断点恢复时还原 Best_Result
    :param extra: 任何额外要保存的字段（如 lr、当前优化器状态、超参数 dict）
    :return: 实际写入的路径
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    payload: Dict[str, Any] = {
        "model_state_dict": model.state_dict(),
        "epoch": epoch,
        "best_score": best_score,
        "save_time": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    if extra:
        # 注意 extra 里若含 tensor，会一并保存
        payload.update(extra)
    torch.save(payload, save_path)
    return save_path


def update_best(model: nn.Module,
                save_dir: str,
                epoch: int,
                current_score: float,
                best_score: float,
                extra: Optional[Dict[str, Any]] = None) -> str:
    """
    若 current_score 优于 best_score，覆盖 best.pt；否则什么都不做。
    返回 best.pt 路径（无论是否更新）。
    """
    best_path = os.path.join(save_dir, "best.pt")
    if current_score >= best_score:
        save_checkpoint(model, best_path, epoch=epoch, best_score=current_score, extra=extra)
    return best_path


def save_last(model: nn.Module,
              save_dir: str,
              epoch: int,
              best_score: Optional[float] = None,
              extra: Optional[Dict[str, Any]] = None) -> str:
    """
    每个 epoch 末覆盖写 last.pt，断电/被 kill 时还能从这里恢复。
    """
    last_path = os.path.join(save_dir, "last.pt")
    save_checkpoint(model, last_path, epoch=epoch, best_score=best_score, extra=extra)
    return last_path


def load_checkpoint(ckpt_path: str, model: nn.Module, map_location: Optional[str] = None) -> Dict[str, Any]:
    """
    从文件加载 state_dict 到 model，并把 ckpt 字典原样返回（含 epoch、best_score 等）。
    """
    ckpt = torch.load(ckpt_path, map_location=map_location)
    if "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        # 兼容旧版仅存 state_dict 的文件
        model.load_state_dict(ckpt)
    return ckpt
