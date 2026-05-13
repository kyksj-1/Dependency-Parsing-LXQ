"""
Scheduler + EMA 单元测试。
"""

import unittest

import torch
import torch.nn as nn

from DataUtils.Scheduler import WarmupCosineSchedule
from DataUtils.EMA import EMAWrapper


class _FakeOptimizer:
    """最小化的假 optimizer,只为测试 scheduler 调 param_groups['lr']"""
    def __init__(self, lrs):
        self.param_groups = [{"lr": lr} for lr in lrs]


class SchedulerTest(unittest.TestCase):
    def test_warmup_linear_increase(self):
        opt = _FakeOptimizer([1.0])
        sched = WarmupCosineSchedule(opt, warmup_steps=10, total_steps=100, min_lr_ratio=0.0)
        lrs = []
        for _ in range(15):
            sched.step()
            lrs.append(opt.param_groups[0]["lr"])
        # step 1 应当是 1/10 = 0.1, step 10 应当是 1.0(峰值)
        self.assertAlmostEqual(lrs[0], 0.1, places=5)
        self.assertAlmostEqual(lrs[9], 1.0, places=5)
        # warmup 后开始下降
        self.assertLess(lrs[14], lrs[9])

    def test_cosine_decay_reaches_min(self):
        opt = _FakeOptimizer([2.0])
        sched = WarmupCosineSchedule(opt, warmup_steps=5, total_steps=20, min_lr_ratio=0.1)
        for _ in range(20):
            sched.step()
        # 最后一步应当接近 min_lr = 2.0 * 0.1 = 0.2
        self.assertAlmostEqual(opt.param_groups[0]["lr"], 0.2, places=3)

    def test_multi_param_groups_independent_base_lr(self):
        # Muon 的混合 param_groups 场景: 不同 group 不同 base_lr
        opt = _FakeOptimizer([0.02, 3e-4])
        sched = WarmupCosineSchedule(opt, warmup_steps=10, total_steps=100, min_lr_ratio=0.0)
        for _ in range(10):
            sched.step()
        # 在峰值 (warmup 末),各 group 应当回到 base_lr
        self.assertAlmostEqual(opt.param_groups[0]["lr"], 0.02, places=5)
        self.assertAlmostEqual(opt.param_groups[1]["lr"], 3e-4, places=5)


class EMATest(unittest.TestCase):
    def test_shadow_initialized_from_model(self):
        m = nn.Linear(8, 4)
        ema = EMAWrapper(m, decay=0.9)
        # shadow 初值应等于 model 权重
        self.assertTrue(torch.allclose(ema.shadow["weight"], m.weight.detach()))
        self.assertTrue(torch.allclose(ema.shadow["bias"], m.bias.detach()))

    def test_update_blends(self):
        m = nn.Linear(4, 2)
        with torch.no_grad():
            m.weight.fill_(2.0)
        ema = EMAWrapper(m, decay=0.5)
        # 改 model 权重为 0,EMA update 后应当 = 0.5*2 + 0.5*0 = 1
        with torch.no_grad():
            m.weight.fill_(0.0)
        ema.update()
        self.assertTrue(torch.allclose(ema.shadow["weight"], torch.tensor(1.0)))

    def test_swap_and_restore(self):
        m = nn.Linear(4, 2)
        with torch.no_grad():
            m.weight.fill_(2.0)
        ema = EMAWrapper(m, decay=0.9)
        # 改 shadow 权重
        with torch.no_grad():
            ema.shadow["weight"].fill_(99.0)

        original = m.weight.detach().clone()
        with ema.swap():
            # 进 context 后 model.weight 应当 = 99
            self.assertTrue(torch.allclose(m.weight, torch.tensor(99.0)))
        # 退出 context 后应当恢复原值
        self.assertTrue(torch.allclose(m.weight, original))


if __name__ == "__main__":
    unittest.main()
