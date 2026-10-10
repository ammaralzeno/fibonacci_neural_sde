"""Experiment 3: joint multi-asset architecture diagram, in the style of the
single-asset white-paper figure. Amber boxes are new vs the single-asset model;
gray boxes are unchanged (now applied per asset)."""
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Circle

from amgm import config as amgm_config

GRAY_FILL, GRAY_EDGE = "#f2f2f2", "#7a7a7a"
BOUNCE, BREAK, HOVER = "#e3a1ab", "#8bb1d9", "#93cb90"
OUT_GREEN, AMBER = "#7cc47f", "#f2c879"
ASSET_COLORS = ["#1f77b4", "#d62728", "#2ca02c"]


def box(ax, x0, y0, x1, y1, fill=GRAY_FILL, r=0.6):
    ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0,
                                boxstyle=f"round,pad=0,rounding_size={r}",
                                fc=fill, ec=GRAY_EDGE, lw=1.2, zorder=2))


def arrow(ax, p0, p1, color="black", lw=1.6, rad=0.0):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=14,
                                 color=color, lw=lw, zorder=3,
                                 connectionstyle=f"arc3,rad={rad}"))


def title(ax, x, y, text):
    ax.text(x, y, text, fontsize=10.5, fontweight="bold", ha="left", va="bottom")


def latent_paths(ax, x0, y0, w, h, n=3):
    rng = np.random.default_rng(0)
    t = np.linspace(0, 1, 40)
    for k in range(n):
        yy = y0 + h * (k + 0.9) / (n + 0.8)
        path = np.cumsum(rng.normal(0, 0.12, 40))
        path = (path - path.min()) / (np.ptp(path) + 1e-9) - 0.5
        ax.plot(x0 + t * w, yy + path * h / (n + 1), color=ASSET_COLORS[k], lw=1.1, zorder=4)


def gate_bars(ax, x, y, label, color):
    for k, ht in enumerate([0.75, 0.5, 0.28]):
        ax.add_patch(Rectangle((x + k * 1.15, y), 0.85, ht * 2.6, fc=color, ec="none", zorder=4))
    ax.text(x + 1.4, y - 0.9, label, fontsize=8, ha="center", va="top")


def expert_stack(ax, x, y, color, label):
    for k in range(3):
        ax.add_patch(FancyBboxPatch((x + 0.7 * k, y + 0.7 * k), 4.2, 1.9,
                                    boxstyle="round,pad=0,rounding_size=0.35",
                                    fc=color, ec=GRAY_EDGE, lw=0.9, zorder=3 + k))
    ax.text(x + 3.6, y - 0.9, label, fontsize=8, ha="center", va="top")


def corr_glyph(ax, x, y, s=1.05):
    vals = [[1.0, 0.8, 0.6], [0.8, 1.0, 0.7], [0.6, 0.7, 1.0]]
    for i in range(3):
        for j in range(3):
            ax.add_patch(Rectangle((x + j * s, y + (2 - i) * s), s, s,
                                   fc=plt.cm.Blues(0.25 + 0.75 * vals[i][j]),
                                   ec="white", lw=0.6, zorder=4))


def moe_box(ax, y0, name, color, pi, gate_lbl):
    box(ax, 47, y0, 67, y0 + 8)
    ax.text(57, y0 + 6.6, f"{name} MoE", fontsize=10, fontweight="bold",
            ha="center", va="center")
    ax.text(57, y0 + 5.3, "per asset $i=1\\dots N$", fontsize=8, style="italic",
            ha="center", va="center", color="#444444")
    gate_bars(ax, 49.2, y0 + 1.6, gate_lbl, color)
    expert_stack(ax, 55.5, y0 + 1.6, color, f"{name}$_1$...{name}$_K$")
    return 67, y0 + 4


def main():
    fig, ax = plt.subplots(figsize=(15.5, 6.8))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 44)
    ax.axis("off")

    # --- inputs -----------------------------------------------------------
    latent_paths(ax, 1, 37.5, 10, 4.5)
    ax.text(1, 35.6, "$X_{t-L:t} = [X^{(1)}, \\dots, X^{(N)}]$", fontsize=10, va="top")
    rng = np.random.default_rng(1)
    for k in range(3):
        x0 = 1 + k * 3.6
        wiggle = np.cumsum(rng.normal(0, 0.15, 25))
        wiggle = (wiggle - wiggle.min()) / (np.ptp(wiggle) + 1e-9)
        ax.plot(x0 + np.linspace(0, 2.6, 25), 19.3 + wiggle * 2.4, color=ASSET_COLORS[k], lw=1.0)
        for lev in [0.25, 0.65]:
            ax.plot([x0, x0 + 2.6], [19.3 + lev * 2.4] * 2, color="#888888", lw=0.7, ls="--")
    ax.text(1, 17.2, "$f_t = [f^{(1)}_t, \\dots, f^{(N)}_t]$  (Fibonacci distances)", fontsize=10, va="top")

    # --- shared encoder + portfolio context -------------------------------
    title(ax, 16, 40.8, "Shared latent path encoder")
    ax.text(16, 39.9, "$Z^{(i)} = \\mathrm{LatentPathEncoder}(X^{(i)})$", fontsize=8.5, va="top", color="#333333")
    box(ax, 16, 32, 30, 40)
    latent_paths(ax, 17.5, 33, 11, 6)

    title(ax, 16, 16.6, "Portfolio context")
    box(ax, 16, 8, 30, 16, fill=AMBER)
    ax.text(23, 12.7, "$c_t = \\frac{1}{N} \\sum_{i=1}^{N} Z^{(i)}$", fontsize=11.5, ha="center", va="center")
    ax.text(23, 10.2, "mean over assets, fed to every gate/expert", fontsize=8, ha="center", va="center", color="#5a4400")

    # --- feature fusion ----------------------------------------------------
    title(ax, 34, 32.6, "Feature fusion (per asset)")
    box(ax, 34, 24, 44, 32)
    ax.text(39, 28, "$h^{(i)} = [Z^{(i)} \\,\\|\\, f^{(i)} \\,\\|\\, c_t]$", fontsize=11, ha="center", va="center")

    # --- per-asset MoE stack ----------------------------------------------
    p_b = moe_box(ax, 34, "Bounce", BOUNCE, None, "Gate $\\pi^B$")
    p_br = moe_box(ax, 24, "Break", BREAK, None, "Gate $\\pi^{Br}$")
    p_h = moe_box(ax, 14, "Hover", HOVER, None, "Gate $\\pi^H$")

    # --- regime FSM ---------------------------------------------------------
    title(ax, 70.5, 32.6, "Regime FSM (per asset)")
    box(ax, 70.5, 24, 79.5, 32)
    for k, (c, yy) in enumerate(zip([BOUNCE, BREAK, HOVER], [30, 28, 26])):
        arrow(ax, (71.5, yy), (74.2, 28), color=c, lw=1.4)
    ax.add_patch(Circle((75, 28), 0.9, fc="white", ec=GRAY_EDGE, lw=1.2, zorder=4))
    arrow(ax, (75.9, 28), (78.2, 28), lw=1.4)

    # --- learned correlation R ----------------------------------------------
    title(ax, 70.5, 16.6, "Learned correlation")
    box(ax, 70.5, 8, 82, 16, fill=AMBER)
    corr_glyph(ax, 72.2, 10.2)
    ax.text(76.6, 13.2, "$R = L_R L_R^{\\top}$", fontsize=11, ha="left", va="center")
    ax.text(76.6, 10.6, "constant, init. from data", fontsize=8, ha="left", va="center", color="#5a4400")

    # --- output ---------------------------------------------------------------
    box(ax, 85.5, 22.5, 99, 34.5, fill=OUT_GREEN, r=0.9)
    ax.text(92.25, 32.4, "Joint update", fontsize=10.5, fontweight="bold", ha="center")
    ax.text(92.25, 30.2, "$X_{t+1} = X_t + \\mu_t\\,\\Delta t + \\mathrm{diag}(\\sigma_t)\\, L_R\\, \\Delta W_t$",
            fontsize=10.5, ha="center", va="center")
    ax.text(92.25, 27.7, "$\\Sigma_t = \\mathrm{diag}(\\sigma_t)\\, R\\, \\mathrm{diag}(\\sigma_t)$",
            fontsize=9.5, ha="center", va="center")
    ax.text(92.25, 25.2, "$X_t, \\mu_t, \\sigma_t \\in \\mathbb{R}^N$", fontsize=8.5, ha="center", va="center", color="#1a4d1a")

    # --- arrows ---------------------------------------------------------------
    arrow(ax, (11.5, 39.5), (15.8, 37), rad=0.0)          # X -> encoder
    arrow(ax, (11.5, 20), (35.5, 23.8), rad=0.12)         # f_t -> fusion
    arrow(ax, (30.2, 36), (36.5, 32.2), rad=-0.18)        # Z -> fusion
    arrow(ax, (30.2, 12), (39, 23.8), rad=-0.12)          # context -> fusion
    for (px, py), rad in zip([p_b, p_br, p_h], [0.25, 0.0, -0.25]):
        arrow(ax, (44.2, 28), (46.8, py), rad=rad)        # fusion -> MoEs
    for (px, py), c, yy, lab_y in zip([p_b, p_br, p_h], [BOUNCE, BREAK, HOVER], [30, 28, 26], [32.6, 28.8, 24.9]):
        arrow(ax, (px, py), (70.3, yy), color=c, rad=-0.1 if py > 28 else (0.1 if py < 28 else 0.0))
        ax.text(68.6, lab_y, "$\\mu, \\sigma$", fontsize=7.5, color=c, ha="center")
    arrow(ax, (79.7, 28), (85.3, 28.6))                   # FSM -> output
    ax.text(82.5, 29.5, "$\\mu_t, \\sigma_t$", fontsize=9, ha="center")
    arrow(ax, (82.2, 12), (88.5, 22.3), rad=-0.15)        # R -> output

    # --- legend -----------------------------------------------------------------
    box(ax, 1, 1.5, 3, 4, fill=AMBER, r=0.3)
    ax.text(3.8, 2.75, "new in the joint model", fontsize=9, va="center")
    box(ax, 17, 1.5, 19, 4, r=0.3)
    ax.text(19.8, 2.75, "unchanged from the single-asset model (now per asset)", fontsize=9, va="center")

    out = amgm_config.work_dir("neural_SDE") / "logs" / "exp3_ablations" / "exp3_architecture.png"
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"written: {out}")


if __name__ == "__main__":
    main()
