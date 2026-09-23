#!/usr/bin/env python3
"""Architecture diagram of the current full network, with every neuronal
projection and its synapse count.

Counts are for the 12%-scale production config (JOB H10 grouped arm:
--dg --ec-lii --ec-lv --mpfc --stc --n-patterns 3 --n-swr 14
--pattern-source ec-lii --schaffer-k 500 --schaffer-group-frac 0.7), taken
from that run's own build log (K x N_post for every fixed_indegree call;
"~" = pairwise_bernoulli expectation). The projection LIST was checked against
a full NEST-kernel census of the same config at 1% scale (every
source->target population pair, GetConnections by source): no projection
exists in the kernel that is missing here, and every fixed_indegree count
there equals K x N_post exactly.

    python plot_architecture.py [--out figures/architecture/network_architecture_12pct.png]
"""
import argparse
import os

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

# ---- populations (12% scale) ------------------------------------------------
POPS = {
    #  name          N        x      y     region
    "EC LII":     (12005,   3.0,  9.0,  "EC"),
    "EC LV":      (7203,    9.0,  9.0,  "EC"),
    "mPFC":       (1440,   14.6,  9.0,  "mPFC"),
    "mPFC INT":   (288,    14.6,  6.9,  "mPFC"),
    "DG GC":      (143990,  2.2,  5.2,  "DG"),
    "DG BSK":     (1190,    0.6,  2.6,  "DG"),
    "MC LOW":     (1785,    2.6,  1.2,  "DG"),
    "MC HIGH":    (1785,    4.2,  2.6,  "DG"),
    "CA3 SUP":    (31675,   7.6,  5.2,  "CA3"),
    "CA3 DEEP":   (7910,    7.6,  1.4,  "CA3"),
    "INT SUP":    (2870,    5.9,  3.3,  "CA3"),
    "INT DEEP":   (945,     9.3,  3.3,  "CA3"),
    "CA1 PYR":    (55195,  13.0,  5.2,  "CA1"),
    "CA1 BSK":    (1680,   14.9,  3.0,  "CA1"),
    "CA1 OLM":    (1085,   11.8,  2.6,  "CA1"),
}

# ---- projections ------------------------------------------------------------
# (src, tgt, synapses, K, kind, rad, label_pos, note)
#   kind: "e" excitatory, "i" inhibitory, "p" plastic excitatory
PROJ = [
    ("CA3 SUP",  "CA1 PYR",  27_597_500, "500",  "e", 0.0,   0.33,  "Schaffer, grouped 0.7"),
    ("CA3 DEEP", "CA1 PYR",   9_217_565, "167",  "e", -0.15, 0.62, "Schaffer"),
    ("CA3 SUP",  "CA1 BSK",     840_000, "500",  "e", -0.28, 0.66, "Schaffer"),
    ("CA3 DEEP", "CA1 BSK",     336_000, "200",  "e", 0.12,  0.7,  "Schaffer"),
    ("DG BSK",   "DG GC",    20_158_600, "140",  "i", 0.15,  0.5,  ""),
    ("EC LII",   "DG GC",     7_199_500, "50",   "e", 0.0,   0.5,  "perforant path"),
    ("CA1 PYR",  "EC LII",      600_250, "50",   "p", -0.35, 0.62,  "STC plastic"),
    ("DG GC",    "CA3 SUP",     475_125, "15",   "e", 0.0,   0.5,  "mossy fibres"),
    ("DG GC",    "CA3 DEEP",     63_280, "8",    "e", 0.15,  0.62, "mossy"),
    ("MC LOW",   "DG GC",       575_960, "4*",   "e", 0.35,  0.5,  ""),
    ("MC HIGH",  "DG GC",       575_960, "4*",   "e", 0.2,   0.55, "silent source"),
    ("DG GC",    "DG BSK",       59_500, "50",   "e", 0.15,  0.5,  ""),
    ("DG GC",    "MC LOW",       53_550, "30",   "e", 0.1,   0.5,  ""),
    ("DG GC",    "MC HIGH",      53_550, "30",   "e", 0.1,   0.5,  ""),
    ("MC LOW",   "DG BSK",       11_900, "10*",  "e", -0.15, 0.5,  ""),
    ("MC HIGH",  "DG BSK",       11_900, "10*",  "e", 0.3,   0.35, ""),
    ("INT SUP",  "CA3 SUP",   4_751_250, "150",  "i", 0.15,  0.5,  ""),
    ("CA3 SUP",  "INT SUP",     143_500, "50",   "e", 0.15,  0.5,  ""),
    ("INT DEEP", "CA3 DEEP",    632_800, "80",   "i", 0.15,  0.5,  ""),
    ("CA3 DEEP", "INT DEEP",     18_900, "20",   "e", 0.15,  0.5,  ""),
    ("INT DEEP", "CA3 SUP",     316_750, "10",   "i", 0.15,  0.5,  ""),
    ("INT SUP",  "CA3 DEEP",     79_100, "10",   "i", 0.15,  0.5,  ""),
    ("CA3 SUP",  "CA3 DEEP",    158_200, "~20",  "e", 0.2,   0.5,   ""),
    ("CA3 DEEP", "CA3 SUP",      31_675, "~1",   "e", 0.2,   0.75,  ""),
    ("CA1 BSK",  "CA1 PYR",   2_759_750, "50",   "i", 0.15,  0.5,  ""),
    ("CA1 OLM",  "CA1 PYR",   1_103_900, "20",   "i", 0.0,   0.5,  ""),
    ("CA1 PYR",  "CA1 BSK",      16_800, "10",   "e", 0.15,  0.5,  ""),
    ("CA1 PYR",  "EC LV",       216_090, "30",   "e", 0.2,   0.5,  ""),
    ("EC LII",   "EC LV",       144_060, "20",   "e", 0.0,   0.5,  ""),
    ("EC LV",    "CA3 SUP",     158_375, "5",    "e", 0.0,   0.5,  ""),
    ("EC LV",    "mPFC",         28_800, "20",   "p", 0.0,   0.5,  "assoc. plastic"),
    ("mPFC",     "mPFC INT",     14_400, "50",   "e", 0.45,   0.5,  ""),
    ("mPFC INT", "mPFC",         17_280, "12",   "i", 0.45,   0.5,  ""),
]

SELF = [
    # (pop, synapses, K, kind, angle_deg)
    ("CA3 SUP",  1_267_000, "~40",  "e", 90),
    ("CA3 DEEP",   102_830, "~13",  "e", 270),
    ("INT SUP",     86_100, "30",   "i", 180),
    ("INT DEEP",    28_350, "30",   "i", 0),
    ("CA1 PYR",    275_975, "5",    "e", 90),
]

EXTERNAL = {
    "EC LII":  "place-field drive\n(28 gen/cell) + bg",
    "DG GC":   "Poisson residual\n(1/cell)",
    "CA3 SUP": "bg + theta/SWR",
    "CA1 PYR": "bg + theta/SWR",
    "CA1 OLM": "theta only —\nno neuronal input",
}

REGION_COL = {"EC": "#e8f1fb", "mPFC": "#f3e9fb", "DG": "#eaf7ea",
              "CA3": "#fdf2e3", "CA1": "#fbe9e9"}
KIND_COL = {"e": "#b2182b", "i": "#2166ac", "p": "#1b7837"}


def fmt(n):
    if n >= 1e6:
        return f"{n/1e6:.1f}M"
    if n >= 1e3:
        return f"{n/1e3:.0f}k"
    return str(n)


def lw(n):
    import math
    return 0.6 + 1.1 * max(0.0, math.log10(n) - 4)


def draw(out):
    fig = plt.figure(figsize=(24, 13))
    ax = fig.add_axes([0.01, 0.03, 0.62, 0.92])
    ax.set_xlim(-0.8, 16.2)
    ax.set_ylim(-0.3, 10.4)
    ax.axis("off")

    # region backgrounds
    regions = {
        "EC": (-0.4, 8.0, 11.1, 2.1),  "mPFC": (13.0, 6.1, 3.1, 4.0),
        "DG": (-0.4, 0.4, 5.6, 6.2),   "CA3": (5.0, 0.5, 5.3, 6.1),
        "CA1": (10.8, 1.8, 5.2, 4.3),
    }
    for r, (x, y, w, h) in regions.items():
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05,rounding_size=0.3",
                                    fc=REGION_COL[r], ec="#999", lw=0.8, zorder=0))
        ax.text(x + 0.15, y + h - 0.12, r, fontsize=15, fontweight="bold",
                va="top", color="#555", zorder=1)

    R = 0.55  # node radius (data units, for arrow shrink)
    for name, (n, x, y, reg) in POPS.items():
        silent = name == "MC HIGH"
        ax.add_patch(FancyBboxPatch((x - 0.72, y - 0.33), 1.44, 0.66,
                                    boxstyle="round,pad=0.02,rounding_size=0.15",
                                    fc="white" if not silent else "#eee",
                                    ec="#333", lw=1.4, ls="--" if silent else "-", zorder=5))
        ax.text(x, y + 0.08, name, ha="center", va="center", fontsize=10.5,
                fontweight="bold", zorder=6)
        ax.text(x, y - 0.17, f"N={n:,}" + (" (silent)" if silent else ""),
                ha="center", va="center", fontsize=8, color="#444", zorder=6)

    for src, tgt, n, k, kind, rad, lp, note in PROJ:
        _, x0, y0, _ = POPS[src]
        _, x1, y1, _ = POPS[tgt]
        style = "-|>" if kind in "ep" else "-["
        arr = FancyArrowPatch((x0, y0), (x1, y1), connectionstyle=f"arc3,rad={rad}",
                              arrowstyle=style + (",head_length=6,head_width=4" if kind in "ep"
                                                  else ",widthB=4,lengthB=2"),
                              color=KIND_COL[kind], lw=lw(n), shrinkA=22, shrinkB=24,
                              ls="--" if kind == "p" else "-", alpha=0.85, zorder=3)
        ax.add_patch(arr)
        # label near the curve: quadratic-bezier point at lp
        mx, my = (x0 + x1) / 2, (y0 + y1) / 2
        dx, dy = x1 - x0, y1 - y0
        cx, cy = mx + rad * dy, my - rad * dx  # arc3 control point
        t = lp
        lx = (1 - t) ** 2 * x0 + 2 * (1 - t) * t * cx + t ** 2 * x1
        ly = (1 - t) ** 2 * y0 + 2 * (1 - t) * t * cy + t ** 2 * y1
        txt = f"{fmt(n)}  K={k}" + (f"\n{note}" if note else "")
        ax.text(lx, ly, txt, fontsize=7.6, ha="center", va="center", color=KIND_COL[kind],
                bbox=dict(fc="white", ec="none", alpha=0.85, pad=0.8), zorder=7)

    import numpy as np
    for pop, n, k, kind, ang in SELF:
        _, x, y, _ = POPS[pop]
        a = np.deg2rad(ang)
        ox, oy = 0.95 * np.cos(a), 0.62 * np.sin(a)
        p0 = (x + 0.35 * np.cos(a + 0.9) * 1.6, y + 0.3 * np.sin(a + 0.9))
        p1 = (x + 0.35 * np.cos(a - 0.9) * 1.6, y + 0.3 * np.sin(a - 0.9))
        ax.add_patch(FancyArrowPatch(p0, p1, connectionstyle="arc3,rad=-2.2",
                                     arrowstyle="-|>,head_length=5,head_width=3" if kind == "e"
                                     else "-[,widthB=3,lengthB=1.5",
                                     color=KIND_COL[kind], lw=lw(n), zorder=4))
        ax.text(x + ox * 1.45, y + oy * 1.55, f"{fmt(n)}\nK={k}", fontsize=7.6,
                ha="center", va="center", color=KIND_COL[kind],
                bbox=dict(fc="white", ec="none", alpha=0.85, pad=0.8), zorder=7)

    for pop, txt in EXTERNAL.items():
        _, x, y, _ = POPS[pop]
        off = {"EC LII": (-2.2, 0.2), "DG GC": (-1.4, 1.1), "CA3 SUP": (0.0, 1.35),
               "CA1 PYR": (0.2, 1.35), "CA1 OLM": (-0.9, -1.0)}[pop]
        ax.annotate(txt, (x, y), (x + off[0], y + off[1]), fontsize=7.5, color="#666",
                    ha="center", va="center", style="italic",
                    arrowprops=dict(arrowstyle="->", color="#aaa", lw=0.8, ls=":"), zorder=2)

    ax.set_title("A   Current network architecture (12% scale, JOB H10 config): "
                 "every neuronal projection, synapse count and in-degree K",
                 loc="left", fontsize=14, fontweight="bold")
    ax.text(-0.6, -0.15,
            "Red = excitatory, blue = inhibitory, green dashed = plastic (STC on CA1→EC LII; "
            "Hebbian association on EC LV→mPFC). Line width ∝ log synapse count. "
            "K* = mossy-cell pool (LOW+HIGH) drawn jointly; split shown ≈50/50.",
            fontsize=9, color="#444")

    # ---- panel B: table ----------------------------------------------------
    bx = fig.add_axes([0.645, 0.03, 0.35, 0.92])
    bx.axis("off")
    rows = [(s, t, n, k, kind) for s, t, n, k, kind, *_ in PROJ]
    rows += [(p, p, n, k, kind) for p, n, k, kind, _ in SELF]
    rows.sort(key=lambda r: -r[2])
    total = sum(r[2] for r in rows)
    bx.set_title("B   All neuronal projections, sorted by synapse count", loc="left",
                 fontsize=14, fontweight="bold")
    y = 0.975
    bx.text(0.00, y, "source → target", fontsize=9.5, fontweight="bold", transform=bx.transAxes)
    bx.text(0.50, y, "K", fontsize=9.5, fontweight="bold", ha="right", transform=bx.transAxes)
    bx.text(0.72, y, "synapses", fontsize=9.5, fontweight="bold", ha="right", transform=bx.transAxes)
    bx.text(0.90, y, "% total", fontsize=9.5, fontweight="bold", ha="right", transform=bx.transAxes)
    dy = 0.925 / (len(rows) + 3)
    for s, t, n, k, kind in rows:
        y -= dy
        c = KIND_COL[kind]
        bx.text(0.00, y, f"{s} → {t}", fontsize=9, color=c, transform=bx.transAxes)
        bx.text(0.50, y, k, fontsize=9, ha="right", color=c, transform=bx.transAxes)
        bx.text(0.72, y, f"{n:,}", fontsize=9, ha="right", color=c, transform=bx.transAxes)
        bx.text(0.90, y, f"{100*n/total:.1f}", fontsize=9, ha="right", color=c, transform=bx.transAxes)
    y -= dy * 1.3
    bx.text(0.00, y, f"Total neuronal synapses: {total:,}  ({len(rows)} projections)",
            fontsize=10, fontweight="bold", transform=bx.transAxes)
    y -= dy * 1.2
    bx.text(0.00, y, "Not counted above: stimulator inputs (Poisson background, theta/SWR sinusoidal\n"
                     "generators, EC LII place-field drive: 1 bg + 28 generators per EC LII cell).",
            fontsize=8.5, color="#555", va="top", transform=bx.transAxes)

    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=130)
    print("saved", out, "total", f"{total:,}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="figures/architecture/network_architecture_12pct.png")
    draw(ap.parse_args().out)
