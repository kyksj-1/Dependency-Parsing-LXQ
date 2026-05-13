"""
Muon Optimizer —— Keller Jordan 2024 提出的"矩阵感知"优化器。

核心思想（一句话）：
    把动量缓冲 G_t 在每一步通过 Newton-Schulz 5 阶迭代正交化为 O_t（≈ 单位奇异值的矩阵），
    然后用 O_t 而不是 G_t 更新参数。这样所有 2D 矩阵参数的"有效学习率"在所有方向上一致。

为什么能 work（Dozat/Adam 视角的直觉）：
    Adam 用 1/√(v) 来 per-coordinate 自适应学习率，但它只在 element 级别上做归一化。
    Muon 直接在"奇异值"级别上做归一化，对 Linear weight 这样的 2D 矩阵更贴合更新本质。
    经验上，Muon 在小 NN 训练（NanoGPT、CIFAR 等）上比 AdamW 同等步数 loss 更低、收敛更快。

实现要点：
    - 只对 2D 非 embedding 矩阵参数走 Muon
    - 1D 参数（bias / LayerNorm scale）+ embedding（稀疏更新）走 AdamW fallback
    - Newton-Schulz 5 阶系数 (a, b, c) = (3.4445, -4.7750, 2.0315) 来自 Keller Jordan 原版
    - 学习率自适应 scaling: max(1, sqrt(fan_out / fan_in))，使非方阵也表现一致

参考实现：https://github.com/KellerJordan/Muon
论文 / 博客：https://kellerjordan.github.io/posts/muon/

接口与 torch.optim.Optimizer 一致：暴露 step / zero_grad / state_dict / load_state_dict / param_groups。
DataUtils/Optim.py 的 Optimizer 类用 methods['Muon'] 接入。
"""

from typing import Iterable, List, Tuple

import torch
from torch.optim.optimizer import Optimizer as TorchOptimizer


# ============================================================================
# Newton-Schulz 5 阶迭代：正交化一个矩阵 G
# ============================================================================
@torch.no_grad()
def zeropower_via_newtonschulz5(G: torch.Tensor, steps: int = 5, eps: float = 1e-7) -> torch.Tensor:
    """
    Newton-Schulz 5 阶迭代，对矩阵 G 做正交化。

    输入: G (M, N) - 任意形状的 2D 矩阵
    输出: O (M, N) - 与 G 同形状，满足 O 的奇异值近似为 1

    数学：
        归一化 X = G / ||G||_F，使谱范数 ≤ 1（前置条件）
        迭代 X <- aX + bA·X + cA²·X，其中 A = XX^T
        系数 (a, b, c) 是 Keller Jordan 调出来的，使得 polynomial p(σ) = aσ + bσ³ + cσ⁵
        在 [0, 1] 区间近似 sign(σ)，迭代 ~5 次即可让所有奇异值约为 1

    Args:
        G:     待正交化的 2D 矩阵
        steps: 迭代次数，默认 5（实测足够）
        eps:   归一化时分母的小常数，防除零
    """
    assert G.ndim == 2, f"Newton-Schulz 仅支持 2D 矩阵, 收到 ndim={G.ndim}"

    # 系数来自 Keller Jordan 原版（kellerjordan.github.io/posts/muon/）
    a, b, c = (3.4445, -4.7750, 2.0315)

    # 用 bfloat16 计算以加速（最终再 cast 回 G.dtype）
    X = G.bfloat16()

    # 让 X 长宽中较短的一边在第 0 维（即 X 的形状是"行 ≤ 列"），矩阵乘更快
    transposed = X.size(0) > X.size(1)
    if transposed:
        X = X.T

    # 归一化到谱范数 ≤ 1
    X = X / (X.norm() + eps)

    for _ in range(steps):
        # A = X X^T （行数较少的一边的 Gram 矩阵）
        A = X @ X.T
        # B = b·A + c·A²
        B = b * A + c * (A @ A)
        # X <- a·X + B·X
        X = a * X + B @ X

    if transposed:
        X = X.T

    return X.to(G.dtype)


# ============================================================================
# Muon optimizer 单体（只处理 2D 参数）
# ============================================================================
class Muon(TorchOptimizer):
    """
    Muon 优化器（Keller Jordan 2024）。只处理 2D 矩阵参数。

    更新规则（per 2D parameter W）:
        buf <- momentum · buf + grad           （动量累积）
        g_eff <- grad + momentum · buf         （Nesterov 等效形式，可选）
        O <- newton_schulz(g_eff, steps=ns_steps)   （正交化）
        scale <- max(1, sqrt(fan_out / fan_in))     （非方阵尺度自适应）
        W <- W - lr · scale · O

    Args:
        params:    待优化的 2D 参数列表（请用 MuonWithAdamW 包装器自动分组）
        lr:        基础学习率，Keller Jordan 默认 0.02
        momentum:  动量系数，默认 0.95
        ns_steps:  Newton-Schulz 迭代次数，默认 5
        nesterov:  是否用 Nesterov 等效形式，默认 True
    """

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 0.02,
        momentum: float = 0.95,
        ns_steps: int = 5,
        nesterov: bool = True,
    ):
        if lr <= 0:
            raise ValueError(f"Invalid lr: {lr}")
        defaults = dict(lr=lr, momentum=momentum, ns_steps=ns_steps, nesterov=nesterov)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            ns_steps = group["ns_steps"]
            nesterov = group["nesterov"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                if g.ndim != 2:
                    raise RuntimeError(
                        f"Muon 仅支持 2D 矩阵参数，收到 ndim={g.ndim}；请用 MuonWithAdamW 包装器把 1D 参数路由到 AdamW"
                    )

                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)
                buf = state["momentum_buffer"]

                # 动量累积: buf <- momentum * buf + g
                buf.mul_(momentum).add_(g)

                # Nesterov 等效: g_eff = g + momentum * buf, 否则 g_eff = buf
                if nesterov:
                    g_eff = g.add(buf, alpha=momentum)
                else:
                    g_eff = buf

                # Newton-Schulz 正交化
                ortho = zeropower_via_newtonschulz5(g_eff, steps=ns_steps)

                # 尺度自适应: 让非方阵参数的有效更新幅度与方阵参数一致
                #   方阵 (M=N): scale=1
                #   非方阵 (M>N or N>M): scale=sqrt(max(M,N)/min(M,N))
                fan_out, fan_in = p.size(0), p.size(1)
                scale = max(1.0, (fan_out / fan_in) ** 0.5)

                # 参数更新
                p.add_(ortho, alpha=-lr * scale)

        return loss


# ============================================================================
# 混合优化器: Muon (2D matrix) + AdamW (1D / embedding)
# ============================================================================
class MuonWithAdamW:
    """
    Hybrid optimizer 包装器。

    为什么需要包装器：
        Muon 只处理 2D 矩阵，但模型里还有 bias / LayerNorm / Embedding 等其他参数。
        Keller Jordan 的实践经验是把这些"非主干"参数 fallback 到 AdamW。
        所以训练时实际跑两个 sub-optimizer，外部看像一个。

    自动分组策略 (from_named_parameters):
        - 名字含 'embed' 的（不论是否 2D） -> AdamW (embedding 是稀疏 lookup,不适合做 NS 正交化)
        - 维度 != 2 的（如 LayerNorm / bias） -> AdamW
        - 其余 2D 矩阵（Linear weight, attention weight） -> Muon

    暴露 step / zero_grad / param_groups / state_dict / load_state_dict 与 PyTorch Optimizer 接口对齐。
    """

    def __init__(
        self,
        muon_params: List[torch.nn.Parameter],
        adamw_params: List[torch.nn.Parameter],
        lr: float = 0.02,
        lr_adamw: float = 3e-4,
        momentum: float = 0.95,
        ns_steps: int = 5,
        nesterov: bool = True,
        adamw_betas: Tuple[float, float] = (0.9, 0.95),
        adamw_eps: float = 1e-8,
        weight_decay: float = 0.0,
    ):
        """
        Args:
            muon_params:    走 Muon 的参数（2D 非 embedding）
            adamw_params:   走 AdamW 的参数（1D 或 embedding）
            lr:             Muon 学习率，Keller Jordan 默认 0.02
            lr_adamw:       AdamW 学习率，常用 3e-4
            momentum:       Muon 动量
            ns_steps:       Newton-Schulz 迭代次数
            nesterov:       Muon 是否用 Nesterov
            adamw_betas:    AdamW (beta1, beta2)
            adamw_eps:      AdamW eps
            weight_decay:   仅作用于 AdamW；Muon 不加 weight decay
                            （Newton-Schulz 已隐式约束矩阵谱范数）
        """
        # 处理空列表的边界情况（极端 case：模型全是 Linear 或全是 embedding）
        self.muon = Muon(
            muon_params if muon_params else [torch.nn.Parameter(torch.zeros(1, 1))],
            lr=lr,
            momentum=momentum,
            ns_steps=ns_steps,
            nesterov=nesterov,
        ) if muon_params else None

        self.adamw = torch.optim.AdamW(
            adamw_params if adamw_params else [torch.nn.Parameter(torch.zeros(1))],
            lr=lr_adamw,
            betas=adamw_betas,
            eps=adamw_eps,
            weight_decay=weight_decay,
        ) if adamw_params else None

        # 把两个子 optimizer 的 param_groups 合并对外暴露
        # （trainer.py 里有 rescale_lrate / set_lrate 会遍历 param_groups）
        self.param_groups: List[dict] = []
        if self.muon is not None:
            self.param_groups.extend(self.muon.param_groups)
        if self.adamw is not None:
            self.param_groups.extend(self.adamw.param_groups)

        # 记录默认参数，便于打印 / 调试
        self.defaults = dict(
            lr=lr, lr_adamw=lr_adamw, momentum=momentum,
            ns_steps=ns_steps, nesterov=nesterov,
            adamw_betas=adamw_betas, adamw_eps=adamw_eps,
            weight_decay=weight_decay,
        )

    @classmethod
    def from_named_parameters(
        cls,
        named_params: List[Tuple[str, torch.nn.Parameter]],
        lr: float = 0.02,
        lr_adamw: float = 3e-4,
        **kwargs,
    ) -> "MuonWithAdamW":
        """
        从 model.named_parameters() 列表自动分组,生成混合优化器。

        路由规则：
            - 名字含 'embed' 的 -> AdamW
            - ndim != 2 的 -> AdamW
            - 其他 2D 矩阵 -> Muon
        """
        muon_params: List[torch.nn.Parameter] = []
        adamw_params: List[torch.nn.Parameter] = []
        muon_names: List[str] = []
        adamw_names: List[str] = []

        for name, p in named_params:
            if not p.requires_grad:
                continue
            if "embed" in name.lower() or p.ndim != 2:
                adamw_params.append(p)
                adamw_names.append(name)
            else:
                muon_params.append(p)
                muon_names.append(name)

        # 打印分组明细，便于调试 / 写报告
        print(f"[Muon分组] Muon (2D non-embed) 共 {len(muon_params)} 个参数:")
        for n in muon_names:
            print(f"    - {n}")
        print(f"[Muon分组] AdamW (1D / embed) 共 {len(adamw_params)} 个参数:")
        for n in adamw_names:
            print(f"    - {n}")

        return cls(
            muon_params=muon_params,
            adamw_params=adamw_params,
            lr=lr,
            lr_adamw=lr_adamw,
            **kwargs,
        )

    # ---------------- Optimizer 接口转发 ----------------
    def step(self, closure=None):
        if self.muon is not None:
            self.muon.step()
        if self.adamw is not None:
            self.adamw.step()

    def zero_grad(self, set_to_none: bool = True):
        if self.muon is not None:
            self.muon.zero_grad(set_to_none=set_to_none)
        if self.adamw is not None:
            self.adamw.zero_grad(set_to_none=set_to_none)

    def state_dict(self):
        return {
            "muon": self.muon.state_dict() if self.muon is not None else None,
            "adamw": self.adamw.state_dict() if self.adamw is not None else None,
        }

    def load_state_dict(self, state_dict):
        if self.muon is not None and state_dict.get("muon") is not None:
            self.muon.load_state_dict(state_dict["muon"])
        if self.adamw is not None and state_dict.get("adamw") is not None:
            self.adamw.load_state_dict(state_dict["adamw"])

    def __repr__(self) -> str:
        return (
            f"MuonWithAdamW(lr={self.defaults['lr']}, "
            f"lr_adamw={self.defaults['lr_adamw']}, "
            f"momentum={self.defaults['momentum']}, "
            f"ns_steps={self.defaults['ns_steps']})"
        )
