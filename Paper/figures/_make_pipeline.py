#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Pipeline 图的 matplotlib fallback 版本。
- 主用 drawio (figures/pipeline.drawio, 手动导出为 pipeline.png 高级感)。
- fallback: 本脚本直接渲染 pipeline.png, 不依赖 drawio, 适合 LaTeX 编译时直接使用。
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
from matplotlib.lines import Line2D

plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial"]
plt.rcParams["axes.unicode_minus"] = False

fig, ax = plt.subplots(figsize=(13.2, 8.4))
ax.set_xlim(0, 1100)
ax.set_ylim(720, 0)        # y 倒置, 0 在顶
ax.axis("off")

PALETTE = {
    "input":    ("#F5F5F5", "#424242"),
    "embed_w":  ("#E3F2FD", "#1565C0"),
    "embed_t":  ("#FFF9C4", "#F9A825"),
    "lex":      ("#FFE0B2", "#E65100"),
    "lstm":     ("#FCE4EC", "#AD1457"),
    "sdpa9":    ("#E8F5E9", "#2E7D32"),
    "large":    ("#F3E5F5", "#6A1B9A"),
    "mlp_arc":  ("#B2EBF2", "#00838F"),
    "mlp_rel":  ("#B2DFDB", "#00695C"),
    "biaf_arc": ("#FFCDD2", "#C62828"),
    "biaf_rel": ("#FFCCBC", "#BF360C"),
    "mst":      ("#D7CCC8", "#4E342E"),
    "loss":     ("#FFE0B2", "#E65100"),
    "loss_t":   ("#FFCCBC", "#BF360C"),
    "opt":      ("#DCEDC8", "#558B2F"),
    "opt_muon": ("#C5E1A5", "#33691E"),
}

def box(xy, wh, text, key, fontsize=10, bold=False, dashed=False, italic=False):
    fc, ec = PALETTE[key]
    x, y = xy; w, h = wh
    rect = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.02,rounding_size=8",
        linewidth=1.5, edgecolor=ec, facecolor=fc,
        linestyle=("dashed" if dashed else "solid"),
    )
    ax.add_patch(rect)
    fw = "bold" if bold else "normal"
    fs = "italic" if italic else "normal"
    ax.text(x + w/2, y + h/2, text, ha="center", va="center",
            fontsize=fontsize, fontweight=fw, fontstyle=fs, color="#222222")
    return (x, y, w, h)

def group_frame(xy, wh, title, color, dashed=True):
    x, y = xy; w, h = wh
    rect = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.02,rounding_size=10",
        linewidth=2, edgecolor=color, facecolor="none",
        linestyle=("dashed" if dashed else "solid"),
    )
    ax.add_patch(rect)
    ax.text(x + w/2, y + 14, title, ha="center", va="center",
            fontsize=11, fontweight="bold", color=color)

def arrow(p1, p2, color="#424242", dashed=False, lw=1.2):
    style = "->,head_length=8,head_width=6"
    arr = FancyArrowPatch(
        p1, p2,
        arrowstyle=style,
        color=color, linewidth=lw,
        linestyle=("dashed" if dashed else "solid"),
        shrinkA=2, shrinkB=2,
    )
    ax.add_patch(arr)

ax.text(550, 22, "Biaffine Dependency Parsing Pipeline",
        ha="center", va="center", fontsize=16, fontweight="bold")

# ============= Embedding Stack =============
in_words = box((40, 70),  (100, 50), "Words\n(B, L)",     "input")
in_tags  = box((160, 70), (100, 50), "POS Tags\n(B, L)",  "input")

emb_w   = box((20, 160),  (80, 50),  "Word Embed\n(learned, 300)",       "embed_w")
emb_ext = box((110, 160), (130, 50), "Pretrained ExtWord\n(frozen, 300)","embed_w", dashed=True)
emb_t   = box((250, 160), (100, 50), "POS Embed\n(learned, 100)",        "embed_t")

# ⊕ symbol
circ = Circle((120, 265), 18, facecolor="white", edgecolor="#424242", linewidth=1.5)
ax.add_patch(circ)
ax.text(120, 265, r"$\oplus$", ha="center", va="center", fontsize=20, fontweight="bold")

lex = box((50, 315), (320, 60),
          "Lexical Feature  [word⊕ext ; tag]\n"
          "dim = 400   |   word-dropout in training",
          "lex", fontsize=10)

# ============= Encoder slot =============
group_frame((40, 410), (340, 180), "Contextual Encoder  (one of three)", "#7B1FA2")
enc_lstm   = box((60, 450),  (95, 55), "BiLSTM 3L\n~13M",                          "lstm")
enc_sdpa9  = box((170, 450), (95, 55), "SDPA-9M\n4L, d=512\nPost-/Pre-LN",         "sdpa9")
enc_large  = box((280, 450), (95, 55), "SDPA-Large\n8L, d=768\nRoPE+SwiGLU+LS",    "large")
ax.text(210, 520, "encoder_type ∈ { lstm, transformer, transformer_large }",
        ha="center", va="center", fontsize=9, color="#555555")
enc_h = box((100, 545), (220, 35),
            r"H ∈ ℝ^{B×L×800}   (common output interface)",
            "lex", fontsize=9)

# ============= MLP projections =============
group_frame((450, 160), (240, 220), "Role-specific MLP Projections", "#00838F")
mlp_arc_dep  = box((470, 200), (90, 50), "MLP\narc-dep",  "mlp_arc")
mlp_arc_head = box((580, 200), (90, 50), "MLP\narc-head", "mlp_arc")
mlp_rel_dep  = box((470, 275), (90, 50), "MLP\nrel-dep",  "mlp_rel")
mlp_rel_head = box((580, 275), (90, 50), "MLP\nrel-head", "mlp_rel")
ax.text(570, 350,
        "dual role: each token is projected separately\n"
        "as a head-role and as a dependent-role vector",
        ha="center", va="center", fontsize=8, color="#555555", style="italic")

# ============= Biaffine =============
group_frame((450, 410), (240, 180), "Biaffine Scorer", "#C62828")
biaf_arc       = box((465, 445), (210, 40),
                     r"$s_{arc}(i\to j) = h_i^\top W_{arc} h_j + b^\top h_j$",
                     "biaf_arc", fontsize=10)
biaf_arc_logit = box((475, 490), (190, 28),
                     r"arc_logit ∈ ℝ^{B×L×L}",
                     "lex", fontsize=9)
biaf_rel       = box((475, 528), (190, 50),
                     "rel_logit ∈ ℝ^{B×L×L×R}\n(trilinear, per-relation)",
                     "biaf_rel", fontsize=9)

# ============= MST + tree =============
mst  = box((750, 475), (160, 55), "MST Decode\n(Chu–Liu–Edmonds)", "mst")
tree = box((930, 475), (150, 55), "Predicted Tree:\n{(head_i, rel_i)}",
           "lex", fontsize=11, bold=True)

# ============= Loss panel =============
group_frame((750, 160), (330, 220), "Training Objective", "#BF360C", dashed=False)
loss_arc = box((765, 200), (300, 35),
               r"$L_{arc} = -\frac{1}{N} \sum_i \log p_{arc}(i \to y^{head}_i)$",
               "loss", fontsize=10)
loss_rel = box((765, 245), (300, 50),
               r"$L_{rel} = -\frac{1}{N} \sum_i \log p_{rel}(i, y^{head}_i, y^{rel}_i)$"
               + "\n(teacher-forced with gold head)",
               "loss", fontsize=10)
loss_total = box((845, 305), (140, 35),
                 r"$L = L_{arc} + L_{rel}$",
                 "loss_t", fontsize=11, bold=True)
ax.text(915, 360, "Eval: UAS (head only) / LAS (head + rel)",
        ha="center", va="center", fontsize=9, color="#555555", style="italic")

# ============= Optimizer panel =============
group_frame((750, 585), (330, 100), "Optimizers compared", "#558B2F", dashed=False)
opt_adam = box((765, 620), (90, 50), "Adam",           "opt")
opt_sgd  = box((865, 620), (90, 50), "SGD\n(control)", "opt")
opt_muon = box((965, 620), (100, 50), "Muon\n(NS-5 ortho)", "opt_muon")

# ============= Arrows =============
def bot(b):  x, y, w, h = b; return (x + w/2, y + h)
def top(b):  x, y, w, h = b; return (x + w/2, y)
def left(b): x, y, w, h = b; return (x,         y + h/2)
def right(b):x, y, w, h = b; return (x + w,     y + h/2)

# in -> emb
arrow(bot(in_words), top(emb_w))
arrow(bot(in_words), top(emb_ext))
arrow(bot(in_tags),  top(emb_t))
# emb -> ⊕ -> lex
arrow(bot(emb_w),   (120, 247))
arrow(bot(emb_ext), (120, 247))
arrow((120, 283),   top(lex))
arrow(bot(emb_t),   (300, 315))
# lex -> encoder
arrow(bot(lex), (210, 410), color="#7B1FA2", lw=2)
# encoder out -> mlp box
arrow(right(enc_h), (450, 270), color="#424242", lw=2)
# mlp -> biaffine
arrow(bot(mlp_arc_dep),  top(biaf_arc), color="#00838F")
arrow(bot(mlp_arc_head), top(biaf_arc), color="#00838F")
arrow(bot(mlp_rel_dep),  top(biaf_rel), color="#00695C")
arrow(bot(mlp_rel_head), top(biaf_rel), color="#00695C")
# biaffine -> mst -> tree
arrow(right(biaf_arc_logit), left(mst), color="#C62828")
arrow(right(mst), left(tree), color="#4E342E")
# biaffine -> loss
arrow((665, 504), (765, 217), color="#E65100", dashed=True)
arrow((665, 553), (765, 270), color="#BF360C", dashed=True)
# loss -> optimizer
arrow(bot(loss_total), (915, 585), color="#558B2F", dashed=True)

plt.tight_layout()
out = "pipeline.png"
plt.savefig(out, dpi=180, bbox_inches="tight", facecolor="white")
print(f"saved -> {out}")
