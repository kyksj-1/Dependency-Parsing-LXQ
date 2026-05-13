"""
TransformerEncoder 单元测试。

不依赖 pytest，仅用 Python 标准库 unittest，方便直接 `python -m unittest tests.test_transformer_encoder` 运行。

运行方式（在 PyTorch_Biaffine_Dependency_Parsing/ 目录下）：
    conda activate research_env
    python -m unittest tests.test_transformer_encoder

或直接：
    python tests/test_transformer_encoder.py

覆盖点：
    1. forward shape 对齐 BiLSTM 输出 (B, L, 2*lstm_hiddens=800)
    2. backward 不报错（loss.backward() 可走通）
    3. mask 生效（padding 位置输出向量被置 0）
    4. 默认参数量在 5–15M 量级（与 BiLSTM 13M 同量级）
    5. 变长 batch 可处理（不同样本不同有效长度）
    6. 训练 / eval 模式切换无副作用（dropout 在 eval 时不应改变结果两次调用）
"""

import unittest

import torch
from Model.Biaffine_Parsing.TransformerEncoder import (
    TransformerEncoder,
    SinusoidalPositionalEncoding,
)


class TransformerEncoderTest(unittest.TestCase):
    # ------------- 测试常量(模拟真实训练规模) ----------------
    BATCH = 4
    SEQ_LEN = 32
    INPUT_SIZE = 200      # = embed_dim(100) + tag_dims(100), Stage 1 默认值
    OUTPUT_DIM = 800      # = 2 * lstm_hiddens(400), Stage 1 baseline 同维度

    def _build_default(self) -> TransformerEncoder:
        """构造一个与 Phase 3 small run 默认配置一致的实例"""
        return TransformerEncoder(
            input_size=self.INPUT_SIZE,
            d_model=512,
            nhead=8,
            num_layers=4,
            ff_size=1024,
            dropout=0.33,
            output_dim=self.OUTPUT_DIM,
        )

    def _build_inputs(self, valid_lens=None):
        """
        构造 (x, masks) 对。
        valid_lens: list[int]，每个样本的有效长度；不传则全部样本满长度。
        """
        x = torch.randn(self.BATCH, self.SEQ_LEN, self.INPUT_SIZE)
        masks = torch.zeros(self.BATCH, self.SEQ_LEN)
        if valid_lens is None:
            masks[:] = 1.0
        else:
            for i, ln in enumerate(valid_lens):
                masks[i, :ln] = 1.0
        return x, masks

    # -------------- 1. shape 对齐 ----------------
    def test_forward_shape_matches_bilstm_output(self):
        encoder = self._build_default()
        x, masks = self._build_inputs()
        out = encoder(x, masks)
        self.assertEqual(
            out.shape,
            (self.BATCH, self.SEQ_LEN, self.OUTPUT_DIM),
            f"输出 shape 应当与 BiLSTM 通路对齐 (B, L, 800),实际 {tuple(out.shape)}",
        )

    # -------------- 2. backward 通路 ----------------
    def test_backward_pass_runs(self):
        encoder = self._build_default()
        encoder.train()
        x, masks = self._build_inputs()
        out = encoder(x, masks)
        # 模拟下游 loss(MLP_arc 后做 CE),这里直接 sum 即可触发反向
        loss = out.sum()
        loss.backward()
        # 检查 input_projection 与 output_projection 都拿到了梯度
        self.assertIsNotNone(encoder.input_projection.weight.grad)
        self.assertIsNotNone(encoder.output_projection.weight.grad)
        # 关键参数梯度不应为 NaN
        for name, param in encoder.named_parameters():
            if param.grad is None:
                continue
            self.assertFalse(
                torch.isnan(param.grad).any().item(),
                f"参数 {name} 梯度出现 NaN",
            )

    # -------------- 3. mask 生效 ----------------
    def test_padding_positions_zeroed(self):
        encoder = self._build_default()
        encoder.eval()  # 关 dropout,确认输出是确定性的
        valid_lens = [self.SEQ_LEN, self.SEQ_LEN - 5, 10, 20]
        x, masks = self._build_inputs(valid_lens=valid_lens)
        with torch.no_grad():
            out = encoder(x, masks)
        # 对每个样本 i,位置 [valid_lens[i]:] 应当全是 0
        for i, ln in enumerate(valid_lens):
            pad_part = out[i, ln:, :]
            self.assertTrue(
                torch.all(pad_part == 0).item(),
                f"样本 {i} padding 区域 [{ln}:] 输出未被 mask 置 0",
            )
            valid_part = out[i, :ln, :]
            # 有效部分大概率不会全 0(被 attention 计算过)
            self.assertFalse(
                torch.all(valid_part == 0).item(),
                f"样本 {i} 有效区域 [:{ln}] 输出全 0,attention 可能没跑",
            )

    # -------------- 4. 参数量量级 ----------------
    def test_param_count_in_expected_range(self):
        encoder = self._build_default()
        n = encoder.num_parameters()
        n_m = n / 1e6
        # 9M 上下浮动允许 5-15M(BiLSTM baseline 13M 同量级)
        self.assertGreater(n_m, 5.0, f"参数太少: {n_m:.2f}M, 应 >5M")
        self.assertLess(n_m, 15.0, f"参数太多: {n_m:.2f}M, 应 <15M(否则不再是 small 配置)")
        print(f"\n[INFO] TransformerEncoder small 配置可训练参数: {n_m:.2f}M")

    # -------------- 5. 变长 batch ----------------
    def test_variable_length_batches(self):
        encoder = self._build_default()
        encoder.eval()
        # 极端:一句满长 + 一句只有 1 个 token
        valid_lens = [self.SEQ_LEN, 1, 16, self.SEQ_LEN - 1]
        x, masks = self._build_inputs(valid_lens=valid_lens)
        with torch.no_grad():
            out = encoder(x, masks)
        self.assertEqual(out.shape, (self.BATCH, self.SEQ_LEN, self.OUTPUT_DIM))
        # 不应出现 NaN
        self.assertFalse(torch.isnan(out).any().item(), "output 含 NaN(可能是注意力被全 mask 掉的样本)")

    # -------------- 6. eval 模式可重复 ----------------
    def test_eval_mode_deterministic(self):
        encoder = self._build_default()
        encoder.eval()
        x, masks = self._build_inputs()
        with torch.no_grad():
            out_a = encoder(x, masks)
            out_b = encoder(x, masks)
        # 关 dropout 后两次调用应当完全一致
        self.assertTrue(
            torch.allclose(out_a, out_b),
            "eval 模式下两次 forward 结果不一致(dropout 没关)",
        )


class SinusoidalPETest(unittest.TestCase):
    def test_pe_shape_and_no_grad(self):
        pe = SinusoidalPositionalEncoding(d_model=64, max_len=100)
        x = torch.zeros(2, 50, 64)
        out = pe(x)
        self.assertEqual(out.shape, (2, 50, 64))
        # PE 是 buffer,不应有 requires_grad
        for buf_name, buf in pe.named_buffers():
            self.assertFalse(buf.requires_grad)


if __name__ == "__main__":
    unittest.main()
