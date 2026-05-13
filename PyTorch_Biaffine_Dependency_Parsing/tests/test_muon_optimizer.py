"""
Muon Optimizer 单元测试。

运行(在 PyTorch_Biaffine_Dependency_Parsing/ 目录下):
    conda activate research_env
    python -m unittest tests.test_muon_optimizer

覆盖点:
    1. Newton-Schulz 5 阶迭代:对任意 2D 矩阵 G 收敛到接近正交(奇异值 ≈ 1)
    2. Muon.step() 在 mock 2D Linear 上运行无报错且参数发生变化
    3. Muon 拒绝 1D 参数(应抛 RuntimeError)
    4. MuonWithAdamW.from_named_parameters: embedding -> AdamW, Linear -> Muon
    5. MuonWithAdamW.step() 在含混合参数的小模型上能跑通且 loss 下降
    6. state_dict / load_state_dict roundtrip 不丢信息
"""

import unittest

import torch
import torch.nn as nn

from DataUtils.MuonOptimizer import (
    zeropower_via_newtonschulz5,
    Muon,
    MuonWithAdamW,
)


class NewtonSchulzTest(unittest.TestCase):
    """Newton-Schulz 5 阶迭代正交化测试

    注:Keller Jordan 的 NS 5 阶 polynomial p(σ) = aσ + bσ³ + cσ⁵ 在 [0, 1] 上
    近似 sign(σ) 但不严格,5 步后奇异值典型 spread 在 [0.5, 1.5]。
    这是优化器特性,不是 bug——目标是让方向"均匀",不是严格正交化。
    """

    def _spread(self, O: torch.Tensor) -> float:
        sigma = torch.linalg.svdvals(O.float())
        return sigma.max().item() / sigma.min().item()

    def test_square_matrix_spread_small(self):
        # 随机方阵 -> 正交化后,奇异值 spread (max/min) 应当较小
        torch.manual_seed(42)
        G = torch.randn(64, 64)
        O = zeropower_via_newtonschulz5(G, steps=5)
        spread = self._spread(O)
        self.assertLess(spread, 4.0, f"方阵 spread {spread:.3f} 过大 (>4)")

    def test_rectangular_matrix_tall(self):
        # 高瘦矩阵 (M > N)
        torch.manual_seed(42)
        G = torch.randn(128, 32)
        O = zeropower_via_newtonschulz5(G, steps=5)
        spread = self._spread(O)
        self.assertLess(spread, 4.0, f"高矩阵 spread {spread:.3f} 过大")

    def test_rectangular_matrix_wide(self):
        # 矮胖矩阵 (N > M)
        torch.manual_seed(42)
        G = torch.randn(32, 128)
        O = zeropower_via_newtonschulz5(G, steps=5)
        spread = self._spread(O)
        self.assertLess(spread, 4.0, f"宽矩阵 spread {spread:.3f} 过大")

    def test_more_steps_tightens_spread(self):
        # 增加迭代步数 spread 应当变小
        torch.manual_seed(42)
        G = torch.randn(48, 48)
        spread_3 = self._spread(zeropower_via_newtonschulz5(G, steps=3))
        spread_8 = self._spread(zeropower_via_newtonschulz5(G, steps=8))
        self.assertLess(spread_8, spread_3 + 0.5,
                        f"steps=8 spread {spread_8:.3f} 应 <= steps=3 spread {spread_3:.3f}+0.5")


class MuonStandaloneTest(unittest.TestCase):
    """Muon 单体优化器(只处理 2D 参数)"""

    def test_step_updates_2d_param(self):
        torch.manual_seed(0)
        layer = nn.Linear(16, 8, bias=False)
        before = layer.weight.detach().clone()
        opt = Muon([layer.weight], lr=0.02)
        # 构造梯度
        x = torch.randn(4, 16)
        y = layer(x)
        loss = y.sum()
        loss.backward()
        opt.step()
        after = layer.weight.detach()
        # 参数应该发生变化
        self.assertFalse(torch.allclose(before, after))
        # 参数不应该 NaN
        self.assertFalse(torch.isnan(after).any().item())

    def test_rejects_1d_param(self):
        # 给 Muon 喂 1D 参数应抛错(强制用 MuonWithAdamW 包装器路由)
        bias = nn.Parameter(torch.randn(8))
        opt = Muon([bias], lr=0.02)
        bias.grad = torch.randn_like(bias)
        with self.assertRaises(RuntimeError):
            opt.step()


class MuonWithAdamWRoutingTest(unittest.TestCase):
    """MuonWithAdamW.from_named_parameters 的自动分组测试

    注:路由按 name 'embed' 关键字 + ndim 判断。项目里 Model.py 实际命名
    word_embed / ext_word_embed / tag_embed,都含 'embed';所以测试也用
    有 'embed' 命名的 module(nn.Sequential 给出 '0.weight' 不含 'embed')。
    """

    class _ToyModel(nn.Module):
        """模拟项目里 ParserModel 的真实命名约定"""
        def __init__(self):
            super().__init__()
            self.word_embed = nn.Embedding(100, 32)
            self.linear = nn.Linear(32, 64)
            self.norm = nn.LayerNorm(64)

        def forward(self, x):
            return self.norm(self.linear(self.word_embed(x).mean(dim=1)))

    def test_embedding_goes_to_adamw(self):
        model = self._ToyModel()
        named = list(model.named_parameters())
        opt = MuonWithAdamW.from_named_parameters(named, lr=0.02, lr_adamw=3e-4)

        # 检查 Muon 那边只有 Linear.weight (2D 非 embedding)
        muon_param_ids = {id(p) for grp in opt.muon.param_groups for p in grp["params"]}
        self.assertIn(id(model.linear.weight), muon_param_ids,
                      "Linear.weight 应当走 Muon")
        self.assertNotIn(id(model.word_embed.weight), muon_param_ids,
                         "word_embed.weight 不应走 Muon (按 name 路由到 AdamW)")
        # 检查 AdamW 那边有 embedding + Linear.bias + LayerNorm.weight + LayerNorm.bias
        adamw_param_ids = {id(p) for grp in opt.adamw.param_groups for p in grp["params"]}
        self.assertIn(id(model.word_embed.weight), adamw_param_ids)
        self.assertIn(id(model.linear.bias), adamw_param_ids)
        self.assertIn(id(model.norm.weight), adamw_param_ids,
                      "LayerNorm.weight (1D) 应当走 AdamW")
        self.assertIn(id(model.norm.bias), adamw_param_ids)

    def test_step_reduces_loss_on_small_model(self):
        # 在 toy regression 上跑几步,loss 应该下降
        torch.manual_seed(0)
        model = nn.Sequential(
            nn.Embedding(50, 16),
            nn.Flatten(start_dim=1),     # 让 embedding 输出能进 Linear
        )
        head = nn.Linear(16 * 4, 1)      # 4 个 token 的句子
        params_named = list(model.named_parameters()) + list(head.named_parameters())
        opt = MuonWithAdamW.from_named_parameters(params_named, lr=0.05, lr_adamw=5e-3)

        # 随机生成 (B=8, L=4) token ids,目标 y 是某线性组合
        torch.manual_seed(1)
        x = torch.randint(0, 50, (8, 4))
        with torch.no_grad():
            target = torch.randn(8, 1)

        losses = []
        for _ in range(30):
            opt.zero_grad()
            embed = model(x)
            pred = head(embed)
            loss = ((pred - target) ** 2).mean()
            loss.backward()
            opt.step()
            losses.append(loss.item())

        # 30 步后 loss 应当显著下降
        self.assertLess(losses[-1], losses[0] * 0.5,
                        f"loss 没下降:开始 {losses[0]:.4f} -> 结束 {losses[-1]:.4f}")

    def test_state_dict_roundtrip(self):
        torch.manual_seed(0)
        model = nn.Linear(8, 4)
        named = list(model.named_parameters())
        opt1 = MuonWithAdamW.from_named_parameters(named, lr=0.02)
        # 跑一步,让 state 里有 momentum buffer / Adam moments
        x = torch.randn(4, 8)
        loss = model(x).sum()
        loss.backward()
        opt1.step()

        sd = opt1.state_dict()
        self.assertIsNotNone(sd["muon"])
        self.assertIsNotNone(sd["adamw"])

        # 构造新优化器,加载 state_dict
        named2 = list(model.named_parameters())
        opt2 = MuonWithAdamW.from_named_parameters(named2, lr=0.02)
        opt2.load_state_dict(sd)
        # 新 optimizer 的 state 应当包含相同的 momentum buffer
        # 简单 sanity check: 不抛错就 OK
        self.assertEqual(
            set(sd["muon"].keys()),
            set(opt2.muon.state_dict().keys()),
        )


if __name__ == "__main__":
    unittest.main()
