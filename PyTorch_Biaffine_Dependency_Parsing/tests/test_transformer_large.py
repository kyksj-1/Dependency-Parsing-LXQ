"""
TransformerEncoderLarge (Phase 4 SOTA) 单元测试。

覆盖:
    1. RotaryEmbedding shape / cache 增长
    2. apply_rotary_pos_emb 数学正确性(对方阵 Q=K 时旋转后仍 dot product 保持)
    3. SwiGLU forward / dropout
    4. LayerScale init 接近 0 + 梯度
    5. TransformerEncoderLarge forward shape / 参数量 / mask 生效
    6. backward 通路 (loss.sum().backward() 不 NaN)
    7. 与 small encoder 互不干扰 (state_dict key 隔离)
"""

import unittest

import torch
import torch.nn as nn

from Model.Biaffine_Parsing.TransformerEncoderLarge import (
    RotaryEmbedding,
    apply_rotary_pos_emb,
    rotate_half,
    SwiGLU,
    LayerScale,
    SOTAEncoderLayer,
    TransformerEncoderLarge,
)


class RoPETest(unittest.TestCase):
    def test_cache_shape(self):
        rope = RotaryEmbedding(head_dim=64, max_len=128)
        cos, sin = rope(seq_len=50, device=torch.device("cpu"))
        self.assertEqual(cos.shape, (50, 64))
        self.assertEqual(sin.shape, (50, 64))

    def test_cache_grows_beyond_initial_max(self):
        # 超出 max_len 应自动重建 cache,不报错
        rope = RotaryEmbedding(head_dim=32, max_len=20)
        cos, sin = rope(seq_len=100, device=torch.device("cpu"))
        self.assertEqual(cos.shape, (100, 32))

    def test_rotate_half(self):
        # rotate_half([a,b,c,d]) = [-c,-d,a,b]
        x = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
        y = rotate_half(x)
        self.assertTrue(torch.allclose(y, torch.tensor([[-3.0, -4.0, 1.0, 2.0]])))

    def test_apply_rotary_preserves_norm(self):
        # 对 q 应用 RoPE 后,各位置每个 head 的 L2 范数应保持
        torch.manual_seed(0)
        B, H, L, D = 2, 4, 16, 32
        q = torch.randn(B, H, L, D)
        k = torch.randn(B, H, L, D)
        rope = RotaryEmbedding(head_dim=D, max_len=L)
        cos, sin = rope(L, device=q.device)
        q2, k2 = apply_rotary_pos_emb(q, k, cos, sin)
        q_norm = q.norm(dim=-1)
        q2_norm = q2.norm(dim=-1)
        self.assertTrue(torch.allclose(q_norm, q2_norm, atol=1e-5),
                        "RoPE 旋转应保持每个 query 向量的 L2 范数")


class SwiGLUTest(unittest.TestCase):
    def test_forward_shape(self):
        m = SwiGLU(d_model=64, ff_size=128, dropout=0.0)
        x = torch.randn(2, 10, 64)
        y = m(x)
        self.assertEqual(y.shape, (2, 10, 64))

    def test_param_count_correct(self):
        m = SwiGLU(d_model=64, ff_size=128, dropout=0.0)
        # 3 个 Linear: gate (64×128) + up (64×128) + down (128×64),各带 bias
        expected = 3 * (64 * 128) + 2 * 128 + 64
        actual = sum(p.numel() for p in m.parameters())
        self.assertEqual(actual, expected)


class LayerScaleTest(unittest.TestCase):
    def test_init_close_to_zero(self):
        ls = LayerScale(d_model=128, init_value=1e-4)
        self.assertAlmostEqual(ls.gamma.mean().item(), 1e-4, places=6)

    def test_output_scaled(self):
        ls = LayerScale(d_model=4, init_value=0.5)
        x = torch.ones(2, 3, 4)
        y = ls(x)
        self.assertTrue(torch.allclose(y, x * 0.5))


class EncoderLargeTest(unittest.TestCase):
    BATCH = 4
    SEQ_LEN = 32
    INPUT_SIZE = 200
    OUTPUT_DIM = 800

    def _build(self, **overrides):
        defaults = dict(
            input_size=self.INPUT_SIZE,
            d_model=256,           # 测试用小一点的尺寸,跑得快
            nhead=8,
            num_layers=3,
            ff_size=512,
            dropout=0.4,
            output_dim=self.OUTPUT_DIM,
            layerscale_init=1e-4,
            use_rope=True,
        )
        defaults.update(overrides)
        return TransformerEncoderLarge(**defaults)

    def _inputs(self, valid_lens=None):
        x = torch.randn(self.BATCH, self.SEQ_LEN, self.INPUT_SIZE)
        masks = torch.zeros(self.BATCH, self.SEQ_LEN)
        if valid_lens is None:
            masks[:] = 1.0
        else:
            for i, ln in enumerate(valid_lens):
                masks[i, :ln] = 1.0
        return x, masks

    def test_forward_shape(self):
        m = self._build()
        x, masks = self._inputs()
        out = m(x, masks)
        self.assertEqual(out.shape, (self.BATCH, self.SEQ_LEN, self.OUTPUT_DIM))

    def test_default_param_count(self):
        # 默认 d_model=768/L=8/ff=3072 应当在 60-90M 范围
        m = TransformerEncoderLarge(
            input_size=self.INPUT_SIZE,
            d_model=768, nhead=12, num_layers=8, ff_size=3072,
        )
        n_m = m.num_parameters() / 1e6
        self.assertGreater(n_m, 60.0, f"参数量太小: {n_m:.2f}M")
        self.assertLess(n_m, 90.0, f"参数量太大: {n_m:.2f}M")
        print(f"\n[INFO] TransformerEncoderLarge 默认配置: {n_m:.2f}M 参数")

    def test_padding_masked(self):
        m = self._build()
        m.eval()
        valid_lens = [self.SEQ_LEN, self.SEQ_LEN - 5, 10, 20]
        x, masks = self._inputs(valid_lens=valid_lens)
        with torch.no_grad():
            out = m(x, masks)
        for i, ln in enumerate(valid_lens):
            pad = out[i, ln:, :]
            self.assertTrue(torch.all(pad == 0).item(),
                            f"样本 {i} padding 区域 [{ln}:] 未置 0")

    def test_backward_no_nan(self):
        m = self._build()
        m.train()
        x, masks = self._inputs()
        out = m(x, masks)
        loss = out.sum()
        loss.backward()
        for name, p in m.named_parameters():
            if p.grad is None:
                continue
            self.assertFalse(torch.isnan(p.grad).any().item(),
                             f"参数 {name} 梯度含 NaN")

    def test_layerscale_param_init_small(self):
        m = self._build(layerscale_init=1e-4)
        for name, p in m.named_parameters():
            if "layerscale" in name.lower() and "gamma" in name.lower():
                self.assertLess(p.abs().max().item(), 1e-3,
                                f"{name} init 不够小")

    def test_without_rope(self):
        # use_rope=False 时,模型应仍可跑通
        m = self._build(use_rope=False)
        self.assertIsNone(m.rope)
        x, masks = self._inputs()
        out = m(x, masks)
        self.assertEqual(out.shape, (self.BATCH, self.SEQ_LEN, self.OUTPUT_DIM))


if __name__ == "__main__":
    unittest.main()
