"""fig_architectures.py — schematic diagrams of the segmentation models used (static, no data needed).
Writes report/figures/arch_*.png. Channel numbers / strides match the configurations in models.py (256² input)."""
from pathlib import Path
import textwrap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = Path(__file__).resolve().parents[1] / "report/figures"; OUT.mkdir(parents=True, exist_ok=True)
C = dict(enc="#3b6ea8", dec="#c8702a", skip="#7a7a7a", att="#b03a5b", misc="#4c8c4a", txt="#1a1a1a", bg="white")
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})


def box(ax, x, y, w, h, label, color, sub=None, fs=8.5):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08", fc=color, ec="none", alpha=.92))
    ax.text(x + w / 2, y + h / 2 + (0.12 if sub else 0), label, ha="center", va="center", color="white", fontsize=fs, weight="bold")
    if sub:
        ax.text(x + w / 2, y + h / 2 - 0.22, sub, ha="center", va="center", color="white", fontsize=7)


def arrow(ax, x0, y0, x1, y1, color="#333", style="-|>", ls="-", rad=0.0, lw=1.2):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle=style, mutation_scale=9, color=color, lw=lw,
                                 linestyle=ls, connectionstyle=f"arc3,rad={rad}"))


def unet_like(fname, title, enc_labels, dec_labels, gates=False, note=""):
    fig, ax = plt.subplots(figsize=(11, 5.2)); ax.set_xlim(0, 11); ax.set_ylim(-0.6, 5.2); ax.axis("off")
    n = len(enc_labels); ys = [4.2 - i * 0.95 for i in range(n)]
    box(ax, 0.1, 4.2, 1.3, 0.7, "Input", C["misc"], "VV, VH, VV−VH")
    for i, (lab, sub) in enumerate(enc_labels):
        box(ax, 1.8 + i * 0.35, ys[i], 1.6, 0.7, lab, C["enc"], sub)
        if i:
            arrow(ax, 1.8 + (i - 1) * 0.35 + 0.8, ys[i - 1], 1.8 + i * 0.35 + 0.8, ys[i] + 0.7)
    arrow(ax, 1.4, 4.55, 1.8, 4.55)
    xd = 7.2
    for i, (lab, sub) in enumerate(dec_labels):
        lvl = n - 2 - i; y = ys[lvl]
        box(ax, xd - lvl * 0.35, y, 1.6, 0.7, lab, C["dec"], sub)
        src_x = 1.8 + lvl * 0.35 + 1.6
        if gates:
            gx = 5.0 - lvl * 0.1
            ax.add_patch(plt.Circle((gx, y + 0.35), 0.22, color=C["att"]))
            ax.text(gx, y + 0.35, "AG", color="white", ha="center", va="center", fontsize=7, weight="bold")
            arrow(ax, src_x, y + 0.35, gx - 0.22, y + 0.35, C["skip"], ls="--")
            arrow(ax, gx + 0.22, y + 0.35, xd - lvl * 0.35, y + 0.35, C["att"])
        else:
            arrow(ax, src_x, y + 0.35, xd - lvl * 0.35, y + 0.35, C["skip"], ls="--")
        prev = (1.8 + (n - 1) * 0.35 + 0.8, ys[n - 1] + 0.7) if i == 0 else (xd - (lvl + 1) * 0.35 + 0.8, ys[lvl + 1] + 0.7)
        arrow(ax, prev[0], prev[1], xd - lvl * 0.35 + 0.8, y, C["dec"])
    box(ax, 9.3, 4.2, 1.5, 0.7, "1×1 conv", C["misc"], "water logit")
    arrow(ax, xd + 1.6, 4.55, 9.3, 4.55)
    ax.text(0.1, -0.3, textwrap.fill(note, 150), fontsize=7.5, color="#444", va="top")
    ax.text(5.5, 5.05, title, ha="center", fontsize=12, weight="bold")
    ax.plot([], [], color=C["skip"], ls="--", label="skip connection"); ax.plot([], [], color=C["dec"], label="upsample ×2 + conv")
    if gates:
        ax.plot([], [], color=C["att"], label="attention-gated skip")
    ax.legend(loc="upper right", bbox_to_anchor=(1.0, 0.62), frameon=False, fontsize=7.5)
    fig.savefig(OUT / fname, dpi=160, bbox_inches="tight", facecolor=C["bg"]); plt.close(fig)


unet_like("arch_unet_resnet.png", "U-Net with ResNet-18/34 encoder (ResU-Net)",
          [("conv1 (stem)", "64 ch, /2"), ("layer1", "64 ch, /4"), ("layer2", "128 ch, /8"), ("layer3", "256 ch, /16"), ("layer4", "512 ch, /32")],
          [("dec 4", "256 ch, /16"), ("dec 3", "128 ch, /8"), ("dec 2", "64 ch, /4"), ("dec 1", "32→16 ch, /1")],
          note="Encoder = ImageNet-pretrained ResNet (residual blocks: 2 conv3×3 + identity shortcut). R18: 14.3 M params, R34: 24.4 M. "
               "Loss: BCE + Dice on valid pixels. Trained on 256² crops, inferred on 512² tiles with 64 px blended overlap.")
unet_like("arch_attention_unet.png", "Attention U-Net (Oktay et al. 2018) — own implementation",
          [("enc 1", "32 ch, /1"), ("enc 2", "64 ch, /2"), ("enc 3", "128 ch, /4"), ("enc 4", "256 ch, /8"), ("bridge", "512 ch, /16")],
          [("dec 4", "256 ch, /8"), ("dec 3", "128 ch, /4"), ("dec 2", "64 ch, /2"), ("dec 1", "32 ch, /1")], gates=True,
          note="Each block: (conv3×3-BN-ReLU)×2, max-pool down, transposed-conv up. Attention gate α = σ(ψ(ReLU(W_g·g + W_x·x))) re-weights the skip "
               "features x with the decoder signal g, suppressing irrelevant regions (e.g. bright urban clutter). 7.9 M params, trained from scratch.")

# attention gate detail
fig, ax = plt.subplots(figsize=(8, 3)); ax.set_xlim(0, 8); ax.set_ylim(0, 3); ax.axis("off")
box(ax, 0.1, 1.9, 1.3, 0.7, "x (skip)", C["enc"], "C×H×W"); box(ax, 0.1, 0.4, 1.3, 0.7, "g (gate)", C["dec"], "C×H×W")
box(ax, 1.9, 1.9, 1.1, 0.7, "W_x 1×1", C["skip"]); box(ax, 1.9, 0.4, 1.1, 0.7, "W_g 1×1", C["skip"])
box(ax, 3.4, 1.15, 0.8, 0.7, "+ ReLU", C["att"]); box(ax, 4.6, 1.15, 1.0, 0.7, "ψ 1×1, σ", C["att"], "α ∈ [0,1]")
box(ax, 6.2, 1.9, 1.6, 0.7, "x ⊙ α", C["misc"], "gated skip")
for a in [(1.4, 2.25, 1.9, 2.25), (1.4, .75, 1.9, .75), (3.0, 2.25, 3.4, 1.7), (3.0, .75, 3.4, 1.3), (4.2, 1.5, 4.6, 1.5), (5.6, 1.6, 6.2, 2.1)]:
    arrow(ax, *a)
arrow(ax, 1.4, 2.5, 6.2, 2.45, C["skip"], ls="--", rad=-0.25)
ax.text(4, 2.9, "Additive attention gate", ha="center", weight="bold", fontsize=11)
fig.savefig(OUT / "arch_attention_gate.png", dpi=160, bbox_inches="tight", facecolor="white"); plt.close(fig)

# DeepLabV3+
fig, ax = plt.subplots(figsize=(11, 3.6)); ax.set_xlim(0, 11); ax.set_ylim(0, 3.6); ax.axis("off")
box(ax, 0.1, 2.3, 1.3, 0.7, "Input", C["misc"], "3 ch")
box(ax, 1.8, 2.3, 1.8, 0.7, "ResNet-34", C["enc"], "dilated, /16")
for i, r in enumerate(["1×1", "3×3 r12", "3×3 r24", "3×3 r36", "img pool"]):
    box(ax, 4.1, 3.0 - i * 0.55, 1.4, 0.45, r, C["att"], fs=7.5)
    arrow(ax, 3.6, 2.65, 4.1, 3.22 - i * 0.55)
box(ax, 5.9, 1.9, 1.3, 0.7, "concat + 1×1", C["att"], "ASPP 256 ch")
box(ax, 7.6, 1.9, 1.5, 0.7, "up ×4, concat", C["dec"], "+ low-level /4")
box(ax, 9.4, 1.9, 1.4, 0.7, "3×3 conv, up ×4", C["misc"], "logit")
arrow(ax, 7.2, 2.25, 7.6, 2.25); arrow(ax, 9.1, 2.25, 9.4, 2.25)
arrow(ax, 2.7, 2.3, 8.3, 1.9, C["skip"], ls="--", rad=0.25)
for i in range(5):
    arrow(ax, 5.5, 3.22 - i * 0.55, 5.9, 2.25)
ax.text(5.5, 3.5, "DeepLabV3+ (ResNet-34): atrous spatial pyramid pooling for multi-scale context", ha="center", weight="bold", fontsize=11)
ax.text(0.1, 0.2, "22.4 M params. Dilated convolutions keep a /16 feature map with a large receptive field; the decoder recovers edges from /4 features.",
        fontsize=7.5, color="#444")
fig.savefig(OUT / "arch_deeplabv3p.png", dpi=160, bbox_inches="tight", facecolor="white"); plt.close(fig)

# SegFormer
fig, ax = plt.subplots(figsize=(11, 3.8)); ax.set_xlim(0, 11); ax.set_ylim(0, 3.8); ax.axis("off")
box(ax, 0.1, 2.5, 1.2, 0.7, "Input", C["misc"], "3 ch")
dims = [("stage 1", "/4, 32|64 ch"), ("stage 2", "/8, 64|128"), ("stage 3", "/16, 160|320"), ("stage 4", "/32, 256|512")]
for i, (l, s) in enumerate(dims):
    box(ax, 1.6 + i * 1.75, 2.5, 1.5, 0.7, l, C["enc"], s)
    if i:
        arrow(ax, 1.6 + (i - 1) * 1.75 + 1.5, 2.85, 1.6 + i * 1.75, 2.85)
    arrow(ax, 1.6 + i * 1.75 + 0.75, 2.5, 5.2, 1.25, C["skip"], ls="--")
arrow(ax, 1.3, 2.85, 1.6, 2.85)
box(ax, 4.4, 0.55, 2.2, 0.7, "all-MLP decoder", C["dec"], "unify ch, up to /4, fuse")
box(ax, 7.2, 0.55, 1.6, 0.7, "MLP → logit", C["misc"], "up ×4")
arrow(ax, 6.6, 0.9, 7.2, 0.9)
ax.text(5.5, 3.65, "SegFormer (MiT-B0 | MiT-B2 encoder)", ha="center", weight="bold", fontsize=11)
ax.text(0.1, 0.1, "Each stage: overlapping patch embedding + efficient self-attention (spatial-reduction K,V) + Mix-FFN (3×3 depth-wise conv, no positional "
        "encoding). B0 3.7 M, B2 24.7 M params. Global context from stage 1 — useful for large contiguous floods; trained with mixed precision.",
        fontsize=7.5, color="#444")
fig.savefig(OUT / "arch_segformer.png", dpi=160, bbox_inches="tight", facecolor="white"); plt.close(fig)
print("figures ->", OUT)
