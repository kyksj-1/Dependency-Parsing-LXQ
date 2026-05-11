"""
Transformer 编码器骨干 —— 用于 Biaffine 依存句法解析器。

设计目标
========
1. 接口与 Layer.MyLSTM 对齐：输入 (B, L, D_in) + mask (B, L) -> 输出 (B, L, output_dim)
2. 用 nn.TransformerEncoder（内部即 Scaled Dot-Product Attention，SDPA）替换 BiLSTM
3. 保持其余整条链路（embedding -> 输入 dropout -> encoder -> MLP -> Biaffine -> MST）不变
   即"只换编码器，做干净的消融"——对照 Stage 1 baseline 即可看出 SDPA vs LSTM 的差异

参数量参考（默认 d_model=512, nhead=8, num_layers=4, ff=1024）
- input projection : 400 × 512                       ≈ 0.20 M
- 4 × encoder layer: 每层 (4×512²) + (2×512×1024)    ≈ 2.10 M / layer × 4 = 8.40 M
- output projection: 512 × 800                       ≈ 0.41 M
- 合计                                                ≈ 9.0  M
（与 BiLSTM baseline 约 13 M 同量级，参数预算公平）

文献参考
========
- Vaswani et al. 2017 (NeurIPS) — Attention Is All You Need（SDPA 原始论文）
- Dozat & Manning 2017 (ICLR)   — Biaffine Parser 原始架构（BiLSTM 版）
- Strubell et al. 2018 / Mrini et al. 2020 / Cui et al. 2022
  —— 把句法解析编码器从 BiLSTM 迁到 Transformer 的代表性后续工作

注意
====
- 本文件只实现"标准 Transformer encoder"。Phase 4 的 SOTA run 会在这之上叠加
  RoPE / LayerScale / Pre-LN / warmup / 更深更宽等技巧，那部分代码届时新增模块。
- 与 MyLSTM 的"variational dropout（同时间步同 mask）"差异：本模块改用 PyTorch
  内置 dropout（attention dropout + FFN dropout），这是 Transformer 的标准做法。
"""

import math
import torch
import torch.nn as nn


class SinusoidalPositionalEncoding(nn.Module):
    """
    经典正弦位置编码（Vaswani et al. 2017）。

    没有可学习参数，通过 register_buffer 注入计算图但不计入 optimizer 更新。
    对依存解析任务（CTB 最长句子约 200 词），max_len=5000 远远够用。
    """

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        # 预先计算 (max_len, d_model) 的位置编码矩阵作为常量缓存
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        # 频率项: exp(-log(10000) * 2i / d_model)，i 从 0 到 d_model/2
        # 偶数维用 sin，奇数维用 cos，构成不同周期的正交基
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float)
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # (1, max_len, d_model) -> 后续广播到 batch 维度
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, L, d_model)
        Returns:
            (B, L, d_model)，位置编码已累加上去
        """
        return x + self.pe[:, : x.size(1)]


class TransformerEncoder(nn.Module):
    """
    Biaffine Parser 的 SDPA 编码器（drop-in 替换 MyLSTM）。

    与 MyLSTM 的接口差异：
    - 输入: (B, L, D_in) + masks (B, L)  ——  与 MyLSTM 对齐（MyLSTM 内部要求 batch_first=True）
    - 输出: (B, L, output_dim)           ——  batch_first；MyLSTM 返回 (L, B, 2*hidden)
                                            两者由 Model.py 用 encoder_type 分支区分
    """

    def __init__(
        self,
        input_size: int,
        d_model: int = 512,
        nhead: int = 8,
        num_layers: int = 4,
        ff_size: int = 1024,
        dropout: float = 0.33,
        output_dim: int = 800,
        max_len: int = 5000,
        norm_first: bool = False,
        activation: str = "gelu",
    ):
        """
        Args:
            input_size: 输入 lexical embedding 维度 (= config.embed_dim + config.tag_dims)，典型 400
            d_model:    Transformer 主干维度（必须能被 nhead 整除）
            nhead:      多头注意力头数
            num_layers: encoder layer 层数
            ff_size:    FFN 中间层维度（标准实践 ≈ 2-4 × d_model）
            dropout:    encoder layer 内部 dropout（含 attn dropout + ffn dropout）
            output_dim: 输出维度，默认 800，与 BiLSTM 双向（hidden=400 × 2）对齐
            max_len:    sinusoidal PE 预计算最大长度
            norm_first: True=Pre-LN（更稳定，不需要 warmup）；False=Post-LN（Vaswani 原版，需 warmup）
                        Phase 3 small 模型用 Post-LN 跑出来作为对比；Phase 4 SOTA 切 Pre-LN
            activation: FFN 激活函数。"gelu" / "relu"，业界目前默认 gelu
        """
        super().__init__()
        assert d_model % nhead == 0, (
            f"d_model ({d_model}) 必须能被 nhead ({nhead}) 整除"
        )

        self.input_size = input_size
        self.d_model = d_model
        self.output_dim = output_dim

        # 1) 输入投影: 把 lexical embedding (D_in=400) 投到 d_model 维度
        self.input_projection = nn.Linear(input_size, d_model)

        # 2) Sinusoidal Positional Encoding (无可学习参数)
        self.positional_encoding = SinusoidalPositionalEncoding(d_model, max_len=max_len)

        # 3) 输入 dropout (PE 加完之后、Transformer 之前)
        self.input_dropout = nn.Dropout(dropout)

        # 4) Transformer encoder 主体
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=ff_size,
            dropout=dropout,
            activation=activation,
            batch_first=True,        # 让 PyTorch 内部按 (B, L, D) 处理，省去手动 transpose
            norm_first=norm_first,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 5) 输出投影: 把 d_model 投回 output_dim，匹配后续 mlp_arc_dep/head 的 input_size
        #    (mlp_arc_* 期望输入 2*lstm_hiddens=800，所以默认 output_dim=800)
        self.output_projection = nn.Linear(d_model, output_dim)

        self._init_parameters()

    def _init_parameters(self) -> None:
        """
        参数初始化策略：
        - Linear 投影层用 Xavier uniform (Vaswani 2017 推荐做法)
        - TransformerEncoderLayer 内部用 PyTorch 默认初始化 (Kaiming-like)，已经合理
        """
        for module in [self.input_projection, self.output_projection]:
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(
        self,
        x: torch.Tensor,
        masks: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x:     (B, L, input_size) lexical embedding (word + tag)
            masks: (B, L) float tensor，1 表示有效 token，0 表示 padding
                   注意 PyTorch TransformerEncoder 的 key_padding_mask 语义相反:
                   True 表示该 key 要被 mask 掉，所以下面要做一次反转

        Returns:
            outputs: (B, L, output_dim)  ——  batch_first，Model.py 不需要再 transpose
        """
        # 1) 投影到 d_model
        h = self.input_projection(x)                     # (B, L, d_model)

        # 2) 加位置编码
        h = self.positional_encoding(h)                  # (B, L, d_model)

        # 3) 输入 dropout
        h = self.input_dropout(h)

        # 4) 构造 key_padding_mask: masks 中为 0 的位置 (padding) 对应 True
        #    PyTorch 约定: key_padding_mask=True 表示该 key 在 attention 中被屏蔽
        key_padding_mask = (masks == 0)                  # (B, L) bool

        # 5) 跑 Transformer encoder
        h = self.encoder(h, src_key_padding_mask=key_padding_mask)  # (B, L, d_model)

        # 6) 把 padding 位置显式置零，避免数值在后续 MLP 中"渗漏"放大
        h = h.masked_fill(key_padding_mask.unsqueeze(-1), 0.0)

        # 7) 投影到 output_dim
        outputs = self.output_projection(h)              # (B, L, output_dim)
        return outputs

    def num_parameters(self) -> int:
        """统计可训练参数总量，便于和 BiLSTM 做参数预算对照"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
