# -*- coding: utf-8 -*-
"""
文件说明：训练循环
本文件相对原版的工程化改造：
    1. 接受外部 logger，所有 print → logger.info；不再用 sys.stdout.write 拼字符串
    2. 引入 History 收集 batch loss / epoch loss / dev/test UAS LAS / lr，训练结束 dump JSON + 画图
    3. 模型保存改用 Checkpoint 模块：每个 epoch 末覆盖 last.pt，best dev 时覆盖 best.pt
       —— 取代旧版按 save_all_model/save_best_model 两条分支重复实现 + 在
       save_dir 复制整个 Model/DataUtils 代码目录的笨重做法
    4. early stop 由原来的 sys.exit() 改为 return，让上层能 dump history、画图后再退
    5. _decay_learning_rate 等死代码删除；学习率动态调整集中到 _dynamic_lr
"""

import os
import time
import random
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.utils as utils

from DataUtils.Optim import Optimizer
from DataUtils.utils import Best_Result, set_lrate
from DataUtils.Common import seed_num, cpu_device
from DataUtils.Logger import History
from DataUtils.Checkpoint import update_best, save_last, save_checkpoint
from Dataloader.DataLoader import batch_variable_depTree
from Dataloader.Dependency import evalDepTree

torch.manual_seed(seed_num)
random.seed(seed_num)


class Train(object):
    """
    Biaffine Parser 的训练协调器。
    使用方式：
        t = Train(train_iter=..., dev_iter=..., test_iter=..., model=parser, config=cfg, logger=logger)
        t.train()
    """

    def __init__(self, **kwargs):
        self.train_iter = kwargs["train_iter"]
        self.dev_iter = kwargs["dev_iter"]
        self.test_iter = kwargs["test_iter"]
        self.parser = kwargs["model"]
        self.config = kwargs["config"]
        # logger 可选；未提供时用 print 兼容
        self.logger = kwargs.get("logger", None)

        self.device = self.config.device
        self.cuda = self.device != cpu_device
        self.early_max_patience = self.config.early_max_patience

        # ---- 优化器 ----
        # SGD/Adam 公用 Optimizer 包装；旧版的 betas=(0.9,0.9) 来自 Dozat 论文，
        # 与 PyTorch 默认 (0.9, 0.999) 不同，对句法依存任务实测更稳定，保留。
        if self.config.learning_algorithm == "SGD":
            self.optimizer = Optimizer(name="SGD", model=self.parser.model,
                                       lr=self.config.learning_rate,
                                       weight_decay=self.config.weight_decay,
                                       grad_clip="None")
        else:
            self.optimizer = Optimizer(name="Adam", model=self.parser.model,
                                       lr=self.config.learning_rate,
                                       weight_decay=self.config.weight_decay,
                                       grad_clip="None",
                                       betas=(0.9, 0.9), eps=1.0e-12)
        self._log("Optimizer: {}".format(self.optimizer))

        # ---- 训练状态 ----
        self.best_score = Best_Result()
        self.train_iter_len = len(self.train_iter)

        # ---- 日志 / 历史 ----
        # save_dir / ckpt_dir 由 main.py 提前创建好，这里直接用
        self.save_dir = self.config.save_dir
        self.ckpt_dir = self.config.ckpt_dir
        self.history = History(save_dir=os.path.join(self.save_dir, "logs"))

    # ---------- helpers ----------
    def _log(self, msg: str) -> None:
        if self.logger is not None:
            self.logger.info(msg)
        else:
            print(msg)

    def _current_lr(self) -> float:
        # Optimizer 包装器把 param_groups 暴露在外层
        return float(self.optimizer.param_groups[0]["lr"])

    def _clip_model_norm(self, clip_use: bool, clip_max_norm) -> None:
        if clip_use:
            gclip = None if clip_max_norm == "None" else float(clip_max_norm)
            assert isinstance(gclip, float)
            utils.clip_grad_norm_(self.parser.model.parameters(), max_norm=gclip)

    def _dynamic_lr(self, epoch: int, new_lr: float) -> float:
        cfg = self.config
        if cfg.use_lr_decay and epoch > cfg.max_patience and (
                epoch - 1) % cfg.max_patience == 0 and new_lr > cfg.min_lrate:
            new_lr = max(new_lr * cfg.lr_rate_decay, cfg.min_lrate)
            set_lrate(self.optimizer, new_lr)
        return new_lr

    def _optimizer_batch_step(self, backward_count: int) -> None:
        cfg = self.config
        if backward_count % cfg.update_batch_size == 0 or backward_count == self.train_iter_len:
            self._clip_model_norm(cfg.clip_max_norm_use, cfg.clip_max_norm)
            self.optimizer.step()
            self.optimizer.zero_grad()

    def _early_stop(self, epoch: int) -> bool:
        """返回 True 表示要早停。"""
        if epoch > self.best_score.best_epoch:
            self.best_score.early_current_patience += 1
            self._log("Dev Has Not Promote {} / {}".format(
                self.best_score.early_current_patience, self.early_max_patience))
            if self.best_score.early_current_patience >= self.early_max_patience:
                self._log("Early Stop. Best dev score on epoch {}.".format(self.best_score.best_epoch))
                return True
        return False

    def _save_ckpt(self, epoch: int, is_best: bool) -> None:
        """
        每 epoch 末统一保存：
            - last.pt：永远覆盖最新（断点恢复用）
            - best.pt：仅当 dev 分数刷新时覆盖
            - epoch_X.pt：仅当 config.save_all_model=True 时额外保存
        """
        extra = {
            "lr": self._current_lr(),
            "config_snapshot": {
                "embed_dim": self.config.embed_dim,
                "lstm_hiddens": self.config.lstm_hiddens,
                "lstm_layers": self.config.lstm_layers,
                "learning_algorithm": self.config.learning_algorithm,
                "learning_rate": self.config.learning_rate,
            },
        }
        save_last(self.parser.model, self.ckpt_dir, epoch=epoch,
                  best_score=self.best_score.best_dev_score, extra=extra)
        if is_best:
            update_best(self.parser.model, self.ckpt_dir, epoch=epoch,
                        current_score=self.best_score.current_dev_score,
                        best_score=self.best_score.best_dev_score, extra=extra)
            self._log("[ckpt] best updated -> {}".format(os.path.join(self.ckpt_dir, "best.pt")))
        if getattr(self.config, "save_all_model", False):
            ep_path = os.path.join(self.ckpt_dir, "epoch_{}.pt".format(epoch))
            save_checkpoint(self.parser.model, ep_path, epoch=epoch,
                            best_score=self.best_score.best_dev_score, extra=extra)

    # ---------- main loop ----------
    def train(self) -> None:
        epochs = self.config.epochs
        new_lr = self.config.learning_rate
        self._log("===== Training Start ===== epochs={}, train_batches={}".format(
            epochs, self.train_iter_len))

        for epoch in range(1, epochs + 1):
            self._log("\n## Epoch {} / {} ##".format(epoch, epochs))
            new_lr = self._dynamic_lr(epoch=epoch, new_lr=new_lr)
            self._log("now lr is {}".format(self._current_lr()))

            start_time = time.time()
            random.shuffle(self.train_iter)
            self.parser.model.train()
            backward_count = 0
            self.optimizer.zero_grad()
            overall_arc_correct = 0
            overall_label_correct = 0
            overall_total_arcs = 0
            epoch_loss_sum = 0.0
            epoch_loss_count = 0

            for batch_count, batch_features in enumerate(self.train_iter):
                backward_count += 1
                words = batch_features.words
                ext_words = batch_features.ext_words
                tags = batch_features.tags
                masks = batch_features.masks
                heads = batch_features.heads
                rels = batch_features.rels
                lengths = batch_features.lengths

                self.parser.forward(words, ext_words, tags, masks)
                loss = self.parser.compute_loss(heads, rels, lengths)
                loss = loss / self.config.update_batch_size
                loss_value = float(loss.detach().cpu().numpy())

                loss.backward()
                self._optimizer_batch_step(backward_count=backward_count)

                # 历史记录
                self.history.add_batch_loss(loss_value)
                epoch_loss_sum += loss_value
                epoch_loss_count += 1

                # 训练侧 UAS/LAS
                if (batch_count) % self.config.log_interval == 0:
                    arc_correct, label_correct, total_arcs = self.parser.compute_accuracy(heads, rels)
                    overall_arc_correct += arc_correct
                    overall_label_correct += label_correct
                    overall_total_arcs += total_arcs
                    if overall_total_arcs > 0:
                        uas = overall_arc_correct.item() * 100.0 / overall_total_arcs
                        las = overall_label_correct.item() * 100.0 / overall_total_arcs
                        self._log("batch [{}/{}] loss={:.6f} sumLen={} ARC={:.2f} REL={:.2f}".format(
                            batch_count + 1, self.train_iter_len,
                            loss_value, sum(lengths), uas, las))

            train_time = time.time() - start_time
            avg_loss = epoch_loss_sum / max(epoch_loss_count, 1)
            train_uas = overall_arc_correct.item() * 100.0 / max(overall_total_arcs, 1)
            train_las = overall_label_correct.item() * 100.0 / max(overall_total_arcs, 1)
            self._log("Train Epoch Time {:.3f}s | avg_loss={:.6f} train_UAS={:.2f} train_LAS={:.2f}".format(
                train_time, avg_loss, train_uas, train_las))

            # 评估
            dev_uas, dev_las = self._eval_batch(self.dev_iter, epoch=epoch, test=False)
            test_uas, test_las = self._eval_batch(self.test_iter, epoch=epoch, test=True)

            # 历史 & ckpt
            is_best = self.best_score.best_test  # _eval_batch 在刷新 best 时设了这个标志
            self.history.add_epoch(epoch_loss=avg_loss,
                                   train_uas=train_uas, train_las=train_las,
                                   dev_uas=dev_uas, dev_las=dev_las,
                                   test_uas=test_uas, test_las=test_las,
                                   lr=self._current_lr())
            self._save_ckpt(epoch=epoch, is_best=is_best)

            # 落盘 + 出图（每 epoch 增量更新，让用户能边训边看）
            self.history.dump()
            self.history.plot()

            # reset best_test 标志（下一个 epoch 重新判断）
            self.best_score.best_test = False

            if self._early_stop(epoch=epoch):
                break

        self._log("===== Training Finished ===== best dev UAS={:.4f} @ epoch {}".format(
            self.best_score.best_dev_score, self.best_score.best_epoch))
        self.history.dump()
        self.history.plot()

    # ---------- eval ----------
    @staticmethod
    def _get_one_batch(insts):
        return [inst.sentence for inst in insts]

    def _eval_batch(self, data_iter, epoch: int, test: bool):
        self.parser.model.eval()
        arc_total_test = 0
        arc_correct_test = 0
        rel_total_test = 0
        rel_correct_test = 0
        alphabet = self.config.alphabet

        eval_start = time.time()
        with torch.no_grad():
            for batch_features in data_iter:
                one_batch = self._get_one_batch(batch_features.insts)
                words = batch_features.words
                ext_words = batch_features.ext_words
                tags = batch_features.tags
                masks = batch_features.masks
                lengths = batch_features.lengths

                arcs_batch, rels_batch = self.parser.parse(words, ext_words, tags, lengths, masks)
                count = 0
                for tree in batch_variable_depTree(one_batch, arcs_batch, rels_batch, lengths, alphabet):
                    arc_total, arc_correct, rel_total, rel_correct = evalDepTree(one_batch[count], tree)
                    arc_total_test += arc_total
                    arc_correct_test += arc_correct
                    rel_total_test += rel_total
                    rel_correct_test += rel_correct
                    count += 1
        eval_time = time.time() - eval_start

        # 防 0 除：评估集若全是被忽略的标签
        uas = arc_correct_test * 100.0 / arc_total_test if arc_total_test > 0 else 0.0
        las = rel_correct_test * 100.0 / rel_total_test if rel_total_test > 0 else 0.0

        flag = "Test" if test else "Dev"
        if not test:
            self.best_score.current_dev_score = uas
            if uas >= self.best_score.best_dev_score:
                self.best_score.best_dev_score = uas
                self.best_score.best_epoch = epoch
                self.best_score.best_test = True
                # dev 刷新时同时清零早停计数器
                self.best_score.early_current_patience = 0
        if test and self.best_score.best_test is True:
            self.best_score.f = uas

        self._log("{} | UAS={}/{}={:.2f} LAS={}/{}={:.2f} | time={:.3f}s".format(
            flag, arc_correct_test, arc_total_test, uas,
            rel_correct_test, rel_total_test, las, eval_time))
        if test:
            self._log("Best dev UAS so far {:.4f} @ epoch {}".format(
                self.best_score.best_dev_score, self.best_score.best_epoch))

        return uas, las
