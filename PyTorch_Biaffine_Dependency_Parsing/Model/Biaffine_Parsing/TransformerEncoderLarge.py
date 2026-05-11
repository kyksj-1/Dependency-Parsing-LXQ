"""
SOTA Transformer Encoder for Biaffine Dependency Parsing —— Phase 4 大参数量 run。

在 Phase 3 small TransformerEncoder 基础上叠加以下"现代 trick":

1. **Pre-LN** (norm_first=True):                                  Xiong et al. 2020
   把 LayerNorm 从 sublayer 后挪到前,梯度更稳,不需要 warmup 也能训
   (本文件仍配合 warmup 用,因为后面还有 RoPE 等加项)

2. **RoPE** (Rotary Positional Embedding):                        Su et al. 2021 (RoFormer)
   把绝对位置编码替换为对 Q/K 做位置相关旋转,相对位置直接由 Q·K 内积体现
   对依存解析这种"长程依赖,但绝对位置不重要"的任务更友好

3. **SwiGLU FFN**:                                                 Shazeer 2020
   FFN 由 Linear-GELU-Linear 改为 SwiGLU: Linear(2d)->silu(half)*half->Linear
   参数效率更高 (LLaMA / PaLM 都用这个)

4. **LayerScale**:                                                 Touvron et al. 2021 (CaiT)
   每个 sublayer 输出乘以可学习 scale,初始化 1e-4
   让深网络初期等价于浅网络,逐步"激活"深度,稳定性好

5. **Dropout 0.4 + Embedding dropout 0.5**:
   70M 参数对 8.3K 句训练集过参数化,需要更强正则

参数量预算
==========
默认 d_model=768 / nhead=12 / num_layers=8 / ff_size=3072:
- input_projection : 400 × 768                       ≈ 0.31 M
- 8 × encoder layer:
    * MHA (Q,K,V,O):    4 × 768²                     = 2.36 M
    * SwiGLU FFN:       768 × (2 × 3072) + 3072 × 768 = 7.08 M
    * LayerNorm × 2:    2 × 768                      ≈ 1.5 K
    * LayerScale × 2:   2 × 768                      ≈ 1.5 K
   subtotal per layer:                               ≈ 9.45 M × 8 = 75.6 M
- output_projection: 768 × 800                       ≈ 0.61 M
- 合计                                                ≈ 76.5 M
(实测会略低,因为 PyTorch 内部 MHA 有 fused 实现)

文献
====
- Vaswani et al. 2017 (Attention Is All You Need)         - 原始 Transformer
- Xiong et al. 2020 (On Layer Normalization in Transformer) - Pre-LN
- Su et al. 2021 (RoFormer)                                - RoPE
- Shazeer 2020 (GLU Variants Improve Transformer)          - SwiGLU
- Touvron et al. 2021 (CaiT)                               - LayerScale
- Polyak & Juditsky 1992                                   - EMA (in trainer)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# 1. RoPE (Rotary Positional Embedding) — Su et al. 2021
# ============================================================================
class RotaryEmbedding(nn.Module):
    """
    对 Q / K 做位置相关旋转的位置编码。

    数学:
        给定 position p 和 head_dim d (偶数), 把 (q_{2i}, q_{2i+1}) 视为复数 q_2i + j q_{2i+1},
        乘以 e^{j p / 10000^{2i/d}} 即在复平面旋转该角度。
        相对位置 (p_q - p_k) 直接体现在 Q · K^T 的内积上。

    与 sinusoidal 区别:
        - sinusoidal: 把 PE 加到 input embedding 上 (绝对位置)
        - RoPE: 把旋转应用到每层的 Q/K (相对位置,且每层独立)
        实测对长程依赖更友好,对绝对位置不敏感的任务 (如句法解析) 收益明显
    """

    def __init__(self, head_dim: int, max_len: int = 4096, base: float = 10000.0):
        super().__init__()
        assert head_dim % 2 == 0, f"RoPE head_dim 必须偶数, 收到 {head_dim}"
        # 频率: 1 / base^(2i / head_dim), i = 0, 1, ..., head_dim/2 - 1
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.max_len = max_len
        self._build_cache(max_len)

    def _build_cache(self, length: int) -> None:
        # positions: (L,) -> frequencies: (L, head_dim/2)
        positions = torch.arange(length, dtype=self.inv_freq.dtype, device=self.inv_freq.device)
        # 外积: (L, head_dim/2)
        freqs = torch.outer(positions, self.inv_freq)
        # 拼成 (L, head_dim): [cos(θ_0), cos(θ_1), ..., cos(θ_{d/2-1}), cos(θ_0), ..., cos(θ_{d/2-1})]
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self, seq_len: int, device: torch.device) -> tuple:
        """返回当前序列长度需要的 cos / sin tensor (L, head_dim)"""
        if seq_len > self.max_len:
            self.max_len = seq_len
            self._build_cache(seq_len)
        return (
            self.cos_cached[:seq_len].to(device),
            self.sin_cached[:seq_len].to(device),
        )


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """把 x 后一半翻负号拼前一半: [-x_{d/2:}, x_{:d/2}]"""
    half = x.size(-1) // 2
    return torch.cat([-x[..., half:], x[..., :half]], dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple:
    """
    对 q, k 应用 RoPE 旋转。

    Args:
        q: (B, nhead, L, head_dim)
        k: (B, nhead, L, head_dim)
        cos / sin: (L, head_dim)
    Returns:
        旋转后的 (q', k')
    """
    # cos / sin: (L, D) -> (1, 1, L, D) 用于广播
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)
    q_rot = (q * cos) + (rotate_half(q) * sin)
    k_rot = (k * cos) + (rotate_half(k) * sin)
    return q_rot, k_rot


# ============================================================================
# 2. SwiGLU FFN — Shazeer 2020
# ============================================================================
class SwiGLU(nn.Module):
    """
    SwiGLU FFN: SiLU 激活 + GLU 门控。

    标准 Transformer FFN:           y = W2(GELU(W1·x))
    SwiGLU FFN:                     y = W3(SiLU(W1·x) * W2·x)

    参数量: 3 × (d_model × ff_size)，比标准 FFN (2 × d_model × ff_size) 多 50%
    但实测表达力增益明显,常用 ff_size = 2/3 × 标准 FFN 大小以保持参数量持平
    """

    def __init__(self, d_model: int, ff_size: int, dropout: float = 0.0):
        super().__init__()
        # 注意:为保持总参数量与 GELU FFN 一致,常设 ff_size = 2/3 × 原值
        # 这里我们让 caller 直接传 ff_size,不做隐式缩放
        self.w_gate = nn.Linear(d_model, ff_size)
        self.w_up = nn.Linear(d_model, ff_size)
        self.w_down = nn.Linear(ff_size, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # SiLU(x) = x * sigmoid(x)
        gated = F.silu(self.w_gate(x)) * self.w_up(x)
        return self.dropout(self.w_down(gated))


# ============================================================================
# 3. LayerScale — Touvron et al. 2021 (CaiT)
# ============================================================================
class LayerScale(nn.Module):
    """
    每个 sublayer 输出乘以可学习 scale γ (shape: (d_model,)),初始化极小 (1e-4)。

    作用:
        - 初期 γ ≈ 0,residual 主导,等价于浅网络,稳定训练
        - γ 慢慢学到合适值,深度逐步"激活"
        - 对深网络 (>6 layer) 显著提升稳定性

    用法:    output = x + layerscale(sublayer(LN(x)))
    """

    def __init__(self, d_model: int, init_value: float = 1e-4):
        super().__init__()
        self.gamma = nn.Parameter(torch.full((d_model,), init_value))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.gamma


# ============================================================================
# 4. SOTA Transformer Encoder Layer (Pre-LN + RoPE + SwiGLU + LayerScale)
# ============================================================================
class SOTAEncoderLayer(nn.Module):
    """
    手写 encoder layer (不用 nn.TransformerEncoderLayer,因为后者不支持 RoPE)。

    结构 (Pre-LN):
        x' = x + LayerScale_1(SelfAttn(LN(x), RoPE))
        x'' = x' + LayerScale_2(SwiGLU(LN(x')))
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        ff_size: int,
        dropout: float = 0.4,
        layerscale_init: float = 1e-4,
        rope: RotaryEmbedding = None,
    ):
        super().__init__()
        assert d_model % nhead == 0
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.rope = rope

        # ---- Self Attention ----
        self.norm1 = nn.LayerNorm(d_model)
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.attn_out_proj = nn.Linear(d_model, d_model)
        self.attn_dropout = nn.Dropout(dropout)
        self.layerscale1 = LayerScale(d_model, init_value=layerscale_init)

        # ---- SwiGLU FFN ----
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = SwiGLU(d_model, ff_size, dropout=dropout)
        self.layerscale2 = LayerScale(d_model, init_value=layerscale_init)

        # ---- residual dropout ----
        self.resid_dropout = nn.Dropout(dropout)

    def _attention(
        self,
        x: torch.Tensor,
        key_padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x: (B, L, d_model) 已经 LN 过
            key_padding_mask: (B, L) bool, True 表示该位置为 padding,需 mask
        Returns:
            (B, L, d_model)
        """
        B, L, D = x.shape
        H = self.nhead
        Dh = self.head_dim

        # (B, L, D) -> (B, nhead, L, head_dim)
        q = self.q_proj(x).view(B, L, H, Dh).transpose(1, 2)
        k = self.k_proj(x).view(B, L, H, Dh).transpose(1, 2)
        v = self.v_proj(x).view(B, L, H, Dh).transpose(1, 2)

        # 应用 RoPE
        if self.rope is not None:
            cos, sin = self.rope(L, device=x.device)
            q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # Scaled dot-product attention (PyTorch 2.x 内置 fused 实现,自动用 FlashAttention)
        # attn_mask 的语义: (B, nhead, L, L) 加性 mask, -inf 表示屏蔽
        # key_padding_mask 转 attn_mask:
        if key_padding_mask is not None:
            # (B, L) -> (B, 1, 1, L), 广播到 (B, nhead, L_q, L_k)
            attn_mask = key_padding_mask[:, None, None, :].to(q.dtype) * -1e4
        else:
            attn_mask = None

        # F.scaled_dot_product_attention 会自动用 FlashAttention(若可用)
        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=self.attn_dropout.p if self.training else 0.0,
            is_causal=False,
        )
        # (B, nhead, L, head_dim) -> (B, L, d_model)
        out = out.transpose(1, 2).contiguous().view(B, L, D)
        out = self.attn_out_proj(out)
        return out

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor) -> torch.Tensor:
        # ---- Pre-LN Self Attention ----
        attn_in = self.norm1(x)
        attn_out = self._attention(attn_in, key_padding_mask)
        x = x + self.resid_dropout(self.layerscale1(attn_out))

        # ---- Pre-LN SwiGLU FFN ----
        ffn_in = self.norm2(x)
        ffn_out = self.ffn(ffn_in)
        x = x + self.resid_dropout(self.layerscale2(ffn_out))

        return x


# ============================================================================
# 5. 主类:TransformerEncoderLarge
# ============================================================================
class TransformerEncoderLarge(nn.Module):
    """
    Phase 4 SOTA encoder. 接口与 Phase 3 TransformerEncoder 对齐:
        forward(x, masks) -> (B, L, output_dim)

    与 Phase 3 small (~9M) 的差异:
        - 参数量: ~75M (用 d_model=768/L=8/ff=3072)
        - Pre-LN (norm_first=True)
        - RoPE 替代 sinusoidal PE
        - SwiGLU FFN 替代 GELU FFN
        - LayerScale (init=1e-4) 每个 sublayer
        - dropout 0.4 (small=0.33)
    """

    def __init__(
        self,
        input_size: int,
        d_model: int = 768,
        nhead: int = 12,
        num_layers: int = 8,
        ff_size: int = 3072,
        dropout: float = 0.4,
        output_dim: int = 800,
        max_len: int = 4096,
        layerscale_init: float = 1e-4,
        use_rope: bool = True,
    ):
        super().__init__()
        assert d_model % nhead == 0
        self.input_size = input_size
        self.d_model = d_model
        self.output_dim = output_dim

        # 1) Input projection
        self.input_projection = nn.Linear(input_size, d_model)
        self.input_dropout = nn.Dropout(dropout)

        # 2) RoPE (实例共享给所有 layer,因为旋转 cache 跟 layer 无关)
        self.rope = RotaryEmbedding(head_dim=d_model // nhead, max_len=max_len) if use_rope else None

        # 3) Encoder layers
        self.layers = nn.ModuleList([
            SOTAEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                ff_size=ff_size,
                dropout=dropout,
                layerscale_init=layerscale_init,
                rope=self.rope,
            )
            for _ in range(num_layers)
        ])

        # 4) 最终 LayerNorm (Pre-LN 范式末尾要补一次 LN)
        self.final_norm = nn.LayerNorm(d_model)

        # 5) Output projection
        self.output_projection = nn.Linear(d_model, output_dim)

        self._init_parameters()

    def _init_parameters(self) -> None:
        # Linear 用 Xavier uniform; LayerNorm / LayerScale 保持各自默认
        for module in [self.input_projection, self.output_projection]:
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        # encoder 内部的 Linear (q_proj/k_proj/v_proj/attn_out_proj/SwiGLU 三个)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                if m.weight is not self.input_projection.weight and m.weight is not self.output_projection.weight:
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:     (B, L, input_size)
            masks: (B, L) float, 1=valid, 0=pad
        Returns:
            (B, L, output_dim)
        """
        # key_padding_mask: True 表示 padding 需 mask (PyTorch 约定)
        key_padding_mask = (masks == 0)  # (B, L)

        h = self.input_projection(x)
        h = self.input_dropout(h)

        for layer in self.layers:
            h = layer(h, key_padding_mask)

        h = self.final_norm(h)

        # padding 位置显式置零(同 Phase 3 行为)
        h = h.masked_fill(key_padding_mask.unsqueeze(-1), 0.0)

        return self.output_projection(h)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
