"""رسم شکل معماری دو مدل MLP و GCN برای گزارش."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

def box(ax, x, y, w, h, text, color):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                fc=color, ec="black", lw=1))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=9.5)

def arrow(ax, x1, x2, y):
    ax.annotate("", xy=(x2, y), xytext=(x1, y), arrowprops=dict(arrowstyle="->", lw=1.3))

fig, axes = plt.subplots(2, 1, figsize=(13, 5.6))
rows = [
    ("MLP  (ignores citations)", [
        ("Input X\n2708 x 1433\n(bag-of-words)", "#dbe9f6"),
        ("Dropout 0.5", "#eeeeee"),
        ("Linear\n1433 -> 16\nH = X W1 + b1", "#fde2c4"),
        ("ReLU\n+ Dropout 0.5", "#eeeeee"),
        ("Linear\n16 -> 7\nZ = H W2 + b2", "#fde2c4"),
        ("Softmax / CE loss\n7 topics", "#d8f0d2"),
    ]),
    ("GCN  (uses citations through  Â = D^-1/2 (A+I) D^-1/2)", [
        ("Input X\n2708 x 1433\n+ graph Â", "#dbe9f6"),
        ("Dropout 0.5", "#eeeeee"),
        ("GCN layer 1\n1433 -> 16\nH = Â X W1 + b1", "#f9c9c9"),
        ("ReLU\n+ Dropout 0.5", "#eeeeee"),
        ("GCN layer 2\n16 -> 7\nZ = Â H W2 + b2", "#f9c9c9"),
        ("Softmax / CE loss\n7 topics", "#d8f0d2"),
    ]),
]
for ax, (title, blocks) in zip(axes, rows):
    ax.set_xlim(0, 13); ax.set_ylim(0, 2); ax.axis("off")
    ax.set_title(title, fontsize=12, loc="left")
    w, gap = 1.75, 0.42
    for i, (t, c) in enumerate(blocks):
        x = 0.1 + i * (w + gap)
        box(ax, x, 0.35, w, 1.3, t, c)
        if i < len(blocks) - 1:
            arrow(ax, x + w + 0.02, x + w + gap - 0.02, 1.0)
plt.tight_layout()
plt.savefig("outputs/figures/00_architecture.png", dpi=150)
