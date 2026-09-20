"""Experiment 2: multi-asset Fibonacci feature analysis.

Extends the single-asset Fibonacci representation to the aligned basket and checks:
(a) levels are meaningful per asset, (b) assets reach levels simultaneously,
(c) events in one asset coincide with movements in another. Writes figures and a
slide-ready findings markdown to the output dir.
"""

import argparse
import importlib
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import ks_2samp

from amgm import config as amgm_config
from amgm.data.neural_SDE_multi import MultiAssetNeuralSDEDataset
from experiments.neural_SDE.fibonacci_state_detection import (
    EventType,
    detect_fibonacci_interactions,
)

FIB_RATIOS = [0.0, 0.236, 0.382, 0.500, 0.618, 0.786, 1.0]
# Paper Section 3.2.1: interaction triggers within a +/-0.02 normalized band.
PAPER_BAND = 0.02
FSM_EVENT_LABELS = {
    EventType.PULLBACK: "Bounce",
    EventType.BREAKOUT: "Break",
    EventType.HOVER: "Hover",
    EventType.TIMEOUT: "Timeout",
}


# ---------------------------------------------------------------------------
# Core computations (importable for tests)
# ---------------------------------------------------------------------------


def levels_original_scale(dataset):
    """Per-window Fibonacci levels denormalized to original price scale: (B, N, 7)."""
    levels = dataset.fib_levels * dataset.sample_range.unsqueeze(-1) + dataset.sample_min.unsqueeze(-1)
    return levels.numpy()


def nearest_level_distances(features):
    """Signed distance to and absolute distance of the nearest Fibonacci level.

    Args:
        features: (B, N, 7) normalized signed distances D = (x - F) / delta.
    Returns:
        signed_nearest: (B, N) signed D of the closest level
        abs_nearest:    (B, N) |D| of the closest level
    """
    features = np.asarray(features)
    abs_d = np.abs(features)
    idx = np.argmin(abs_d, axis=-1)
    signed_nearest = np.take_along_axis(features, idx[..., None], axis=-1)[..., 0]
    abs_nearest = np.take_along_axis(abs_d, idx[..., None], axis=-1)[..., 0]
    return signed_nearest, abs_nearest


def near_level_mask(features, eps=PAPER_BAND):
    """(B, N) boolean: asset is within eps (normalized) of its nearest level."""
    _, abs_nearest = nearest_level_distances(features)
    return abs_nearest < eps


def next_day_returns(dataset):
    """Real next-day simple returns aligned to each window: (B, N).

    Window s ends at dates[s + w - 1]; the target day is s + w.
    """
    prices = dataset.prices_aligned
    w = dataset.lookback_window
    n_windows = len(dataset)
    end_idx = np.arange(n_windows) + w - 1
    rets = np.full((n_windows, prices.shape[1]), np.nan, dtype=np.float64)
    valid = end_idx + 1 < prices.shape[0]
    rets[valid] = prices[end_idx[valid] + 1] / prices[end_idx[valid]] - 1.0
    return rets


def pairwise_cooccurrence(mask):
    """Co-occurrence of near-level events between asset pairs.

    Returns dict with:
        p_near: (N,) marginal P(asset near a level)
        joint:  (N, N) P(i near and j near)
        lift:   (N, N) joint / (p_i * p_j); lift > 1 means the pair reaches
                levels together more than independence implies. NaN where
                a marginal rate is 0.
    """
    mask = np.asarray(mask, dtype=bool)
    n = mask.shape[1]
    p_near = mask.mean(axis=0)
    joint = (mask[:, :, None] & mask[:, None, :]).mean(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        lift = joint / np.outer(p_near, p_near)
    lift[p_near == 0, :] = np.nan
    lift[:, p_near == 0] = np.nan
    return {"p_near": p_near, "joint": joint, "lift": lift, "n_assets": n}


def lagged_indicator_xcorr(mask, max_lag=10):
    """Pearson cross-correlation of near-level indicators at lags -max_lag..max_lag.

    xcorr[k, i, j] = corr(near_i(t), near_j(t + lag_k)); positive lag means
    asset j's level approaches lag asset i's. Pairs with a constant series
    (no variance) give NaN.
    """
    mask = np.asarray(mask, dtype=float)
    T, n = mask.shape
    lags = np.arange(-max_lag, max_lag + 1)
    xcorr = np.full((len(lags), n, n), np.nan)
    for k, lag in enumerate(lags):
        for i in range(n):
            for j in range(n):
                a = mask[max(0, -lag) : T - max(0, lag), i]
                b = mask[max(0, lag) : T - max(0, -lag), j]
                if a.size < 10 or a.std() == 0 or b.std() == 0:
                    continue
                xcorr[k, i, j] = np.corrcoef(a, b)[0, 1]
    return lags, xcorr


def conditional_returns(mask, rets):
    """Asset B's next-day returns conditioned on asset A being near a level.

    Returns an (N, N) dict of ordered pairs (A, B) -> stats; diagonal and
    pairs with too few conditional samples hold NaN stats.
    """
    mask = np.asarray(mask, dtype=bool)
    rets = np.asarray(rets, dtype=float)
    n = mask.shape[1]
    out = {}
    for a in range(n):
        for b in range(n):
            cond = rets[mask[:, a], b]
            uncond = rets[~mask[:, a], b]
            cond = cond[np.isfinite(cond)]
            uncond = uncond[np.isfinite(uncond)]
            stats = {
                "n_cond": int(cond.size),
                "mean_abs_cond": float(np.abs(cond).mean()) if cond.size else np.nan,
                "mean_abs_uncond": float(np.abs(uncond).mean()) if uncond.size else np.nan,
                "ks_stat": np.nan,
                "ks_p": np.nan,
                "cond": cond,
                "uncond": uncond,
            }
            if a != b and cond.size >= 10 and uncond.size >= 10:
                ks = ks_2samp(cond, uncond)
                stats["ks_stat"] = float(ks.statistic)
                stats["ks_p"] = float(ks.pvalue)
            out[(a, b)] = stats
    return out


def fsm_events_per_asset(prices_aligned, tolerance=PAPER_BAND):
    """Run the Fibonacci interaction FSM per asset on the full aligned series.

    Prices are min-max normalized per asset over the whole period so the
    tolerance matches the paper's normalized +/-0.02 band; levels are the seven
    ratios of that period's range (static, as the detector expects).
    Returns (N,) list of dicts mapping event label -> count.
    """
    prices = np.asarray(prices_aligned, dtype=float)
    n = prices.shape[1]
    counts = []
    for a in range(n):
        p = prices[:, a]
        p_min, p_max = p.min(), p.max()
        rng = max(p_max - p_min, 1e-8)
        p_norm = (p - p_min) / rng
        events = detect_fibonacci_interactions(p_norm, FIB_RATIOS, tolerance=tolerance)
        c = {label: 0 for label in FSM_EVENT_LABELS.values()}
        for ev in events:
            c[FSM_EVENT_LABELS[ev["event_type"]]] += 1
        counts.append(c)
    return counts


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def _date_ticks(ax, dates, n_ticks=6):
    """Sparse date tick labels on an integer-indexed time axis."""
    idx = np.unique(np.linspace(0, len(dates) - 1, n_ticks).round().astype(int))
    ax.set_xticks(idx)
    ax.set_xticklabels([dates[i] for i in idx], rotation=45, ha="right", fontsize=7)


def plot_simultaneous_levels(dates, prices, levels, basket, output_file, segment_days=252):
    """Fig A: stacked per-asset panels of price + rolling Fibonacci levels, shared time axis."""
    n = len(basket)
    seg = min(segment_days, len(dates))
    dates, prices, levels = dates[-seg:], prices[-seg:], levels[-seg:]
    colors = plt.cm.tab10.colors

    fig, axes = plt.subplots(n, 1, figsize=(11, 2.2 * n), sharex=True)
    axes = np.atleast_1d(axes)
    for a, ax in enumerate(axes):
        ax.plot(prices[:, a], color="black", linewidth=1.4, label="price")
        for k in range(levels.shape[-1]):
            ax.plot(levels[:, a, k], color=colors[k % len(colors)], linestyle="--", linewidth=0.9,
                    label=f"F {FIB_RATIOS[k]}")
        ax.set_ylabel(basket[a], fontsize=9)
        ax.grid(True, alpha=0.3)
        if a == 0:
            ax.legend(fontsize=7, ncol=4, loc="best")
    _date_ticks(axes[-1], dates)
    fig.suptitle("Multi-asset Fibonacci levels (rolling 252d window), aligned calendar", fontsize=12)
    fig.tight_layout()
    fig.savefig(output_file, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_distance_heatmap(dates, abs_nearest, basket, eps, output_file):
    """Fig B: time x asset heatmap of distance to the nearest level (near-level = dark)."""
    fig, ax = plt.subplots(figsize=(11, 3.6))
    im = ax.imshow(
        abs_nearest.T, aspect="auto", cmap="viridis_r", vmin=0.0,
        vmax=max(0.12, float(np.nanmax(abs_nearest))),
    )
    ax.set_yticks(range(len(basket)), basket, fontsize=8)
    _date_ticks(ax, dates)
    fig.colorbar(im, ax=ax, fraction=0.025, label="min |D| (normalized)")
    # Mark near-level days
    near_t, near_a = np.where(abs_nearest < eps)
    ax.scatter(near_t, near_a, s=2, color="red", label=f"|D| < {eps}")
    ax.legend(fontsize=8, loc="upper right")
    ax.set_title("Distance to nearest Fibonacci level per asset (red = near-level days)", fontsize=11)
    fig.tight_layout()
    fig.savefig(output_file, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_signed_distance_histograms(signed_nearest, basket, output_file):
    """Check (a): per-asset histograms of signed distance to the nearest level."""
    n = len(basket)
    fig, axes = plt.subplots(1, n, figsize=(3.0 * n, 3.2), sharey=True)
    axes = np.atleast_1d(axes)
    for a, ax in enumerate(axes):
        ax.hist(signed_nearest[:, a], bins=40, color="tab:blue", alpha=0.8)
        ax.axvline(0.0, color="red", linestyle="--", linewidth=1.2)
        ax.set_title(basket[a], fontsize=9)
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("window count")
    fig.suptitle("Signed distance to nearest Fibonacci level at window end (0 = on a level)", fontsize=11)
    fig.tight_layout()
    fig.savefig(output_file, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_fsm_events(counts, basket, output_file):
    """Check (a): per-asset FSM event counts (Bounce/Break/Hover/Timeout) on real data."""
    labels = ["Bounce", "Break", "Hover", "Timeout"]
    n = len(basket)
    x = np.arange(len(labels))
    width = 0.8 / n
    fig, ax = plt.subplots(figsize=(8.5, 4))
    for a in range(n):
        ax.bar(x + a * width, [counts[a][lab] for lab in labels], width=width, label=basket[a])
    ax.set_xticks(x + width * (n - 1) / 2, labels)
    ax.set_ylabel("event count")
    ax.set_title("Fibonacci interaction events per asset (real data, FSM detector)", fontsize=11)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(output_file, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_cooccurrence_heatmap(lift, p_near, basket, eps, output_file):
    """Check (b): lift of joint near-level events vs independence (diagonal masked)."""
    n = len(basket)
    shown = lift.copy()
    np.fill_diagonal(shown, np.nan)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))

    im0 = axes[0].imshow(shown, cmap="RdBu_r", vmin=np.nanmin(shown), vmax=np.nanmax(shown))
    axes[0].set_title(f"Near-level co-occurrence lift (eps={eps})\n>1 = reach levels together more than chance", fontsize=10)
    fig.colorbar(im0, ax=axes[0], fraction=0.046)
    for ax in axes[:1]:
        ax.set_xticks(range(n), basket, rotation=90, fontsize=8)
        ax.set_yticks(range(n), basket, fontsize=8)
    for i in range(n):
        for j in range(n):
            if i != j and np.isfinite(shown[i, j]):
                axes[0].text(j, i, f"{shown[i, j]:.2f}", ha="center", va="center", fontsize=8)

    axb = axes[1]
    axb.bar(range(n), p_near, color="tab:blue")
    axb.set_xticks(range(n), basket, rotation=90, fontsize=8)
    axb.set_ylabel("P(near level)")
    axb.set_title(f"Marginal near-level rate per asset (eps={eps})", fontsize=10)
    axb.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(output_file, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_lagged_xcorr(lags, xcorr, output_file):
    """Check (b): mean off-diagonal cross-correlation of near-level indicators vs lag."""
    n = xcorr.shape[1]
    off = ~np.eye(n, dtype=bool)
    mean_x = np.nanmean(xcorr[:, off].reshape(len(lags), -1), axis=1)
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    ax.bar(lags, mean_x, color="tab:blue")
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("lag (days); >0 means asset B's approach lags asset A's")
    ax.set_ylabel("mean pairwise corr of near-level indicators")
    ax.set_title("Lead/lag of Fibonacci level approaches across assets", fontsize=11)
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(output_file, dpi=150, bbox_inches="tight")
    plt.close(fig)
    peak = int(np.nanargmax(np.abs(mean_x)))
    return {"peak_lag": int(lags[peak]), "peak_abs_xcorr": float(mean_x[peak])}


def plot_conditional_returns(cond_stats, basket, output_heatmap, output_overlays, top_k=4):
    """Check (c): KS heatmap over ordered pairs + return histograms for the top-K pairs."""
    n = len(basket)
    ks_mat = np.full((n, n), np.nan)
    for (a, b), s in cond_stats.items():
        ks_mat[a, b] = s["ks_stat"]

    fig, ax = plt.subplots(figsize=(6.5, 5))
    im = ax.imshow(ks_mat, cmap="viridis", vmin=0, vmax=np.nanmax(ks_mat) if np.isfinite(ks_mat).any() else 1)
    ax.set_title("KS statistic: B's next-day returns | A near level\nvs A not near (row = A, col = B)", fontsize=10)
    ax.set_xticks(range(n), basket, rotation=90, fontsize=8)
    ax.set_yticks(range(n), basket, fontsize=8)
    for i in range(n):
        for j in range(n):
            if np.isfinite(ks_mat[i, j]):
                ax.text(j, i, f"{ks_mat[i, j]:.2f}", ha="center", va="center", fontsize=8, color="white")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(output_heatmap, dpi=150, bbox_inches="tight")
    plt.close(fig)

    ranked = sorted(
        ((s["ks_stat"], (a, b)) for (a, b), s in cond_stats.items() if np.isfinite(s["ks_stat"])),
        reverse=True,
    )[:top_k]
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    if ranked:
        for ax, (ks_val, (a, b)) in zip(axes.flat, ranked):
            s = cond_stats[(a, b)]
            bins = np.linspace(
                min(s["cond"].min(), s["uncond"].min()),
                max(s["cond"].max(), s["uncond"].max()),
                40,
            )
            ax.hist(s["uncond"], bins=bins, density=True, alpha=0.6, label=f"{basket[b]} | {basket[a]} NOT near", color="gray")
            ax.hist(s["cond"], bins=bins, density=True, alpha=0.6, label=f"{basket[b]} | {basket[a]} near level", color="tab:red")
            ax.set_title(f"KS={ks_val:.2f}, n={s['n_cond']}", fontsize=9)
            ax.legend(fontsize=7)
            ax.grid(True, alpha=0.3)
        for ax in list(axes.flat)[len(ranked):]:
            ax.axis("off")
    else:
        axes.flat[0].text(0.5, 0.5, "no pairs with enough conditional samples",
                          ha="center", va="center", transform=axes.flat[0].transAxes)
    fig.suptitle("Conditional next-day return distributions (top KS pairs)", fontsize=11)
    fig.tight_layout()
    fig.savefig(output_overlays, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return [(a, b, float(k)) for k, (a, b) in ranked]


# ---------------------------------------------------------------------------
# Findings markdown
# ---------------------------------------------------------------------------


def write_findings(output_dir, basket, dates, mask, cooc, xcorr_summary, fsm_counts, cond_stats, top_pairs, eps):
    """Slide-ready Exp 2 findings with the key numbers and figure references."""
    n = len(basket)
    off = ~np.eye(n, dtype=bool)
    lift_off = cooc["lift"][off]
    lift_off = lift_off[np.isfinite(lift_off)]

    lines = [
        "# Experiment 2 — Multi-asset Fibonacci features: findings",
        "",
        f"Basket ({n} assets): {', '.join(basket)}",
        f"Aligned calendar: {dates[0]} .. {dates[-1]}; rolling 252d windows: {mask.shape[0]} window-ends.",
        f"Near-level definition: |D| < {eps} (paper's +/-0.02 band, normalized by window range).",
        "",
        "## (a) Are Fibonacci levels meaningful independently for each asset?",
        "",
        f"- Near-level rate per asset: " + ", ".join(f"{b}: {p:.1%}" for b, p in zip(basket, cooc["p_near"])),
        "- FSM interaction events on real data (Bounce/Break/Hover/Timeout per asset):",
    ]
    for b, c in zip(basket, fsm_counts):
        lines.append(f"  - {b}: " + ", ".join(f"{k}={v}" for k, v in c.items()))
    lines += [
        "",
        "Figures: `figA_simultaneous_levels.png`, `figB_distance_heatmap.png`, "
        "`check_a_signed_distance_hist.png`, `check_a_fsm_events.png`.",
        "",
        "## (b) Do different assets reach Fibonacci levels simultaneously?",
        "",
        f"- Mean pairwise co-occurrence lift: {lift_off.mean():.2f} "
        f"(1.0 = independence; >1 = simultaneous more often than chance).",
        f"- Max lift: {np.nanmax(lift_off):.2f}; min lift: {np.nanmin(lift_off):.2f}.",
        f"- Lead/lag: mean pairwise indicator cross-correlation peaks at lag "
        f"{xcorr_summary['peak_lag']} days (|corr|={abs(xcorr_summary['peak_abs_xcorr']):.3f}).",
        "",
        "Figures: `check_b_cooccurrence_lift.png`, `check_b_lagged_xcorr.png`.",
        "",
        "## (c) Do Fibonacci events in one asset coincide with movements in another?",
        "",
        "Ordered pairs (A near level -> B's next-day returns), top KS statistic pairs:",
    ]
    for a, b, ks_val in top_pairs:
        s = cond_stats[(a, b)]
        lines.append(
            f"- {basket[a]} -> {basket[b]}: KS={ks_val:.2f} (p={s['ks_p']:.3f}), n={s['n_cond']}, "
            f"mean|ret| {s['mean_abs_cond']:.4f} vs {s['mean_abs_uncond']:.4f} unconditional"
        )
    lines += [
        "",
        "Figures: `check_c_conditional_ks_heatmap.png`, `check_c_conditional_overlays.png`.",
    ]
    (Path(output_dir) / "exp2_findings.md").write_text("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main_fib_analysis(
    dataset,
    output_dir=None,
    eps=PAPER_BAND,
    segment_days=252,
    max_lag=10,
):
    """Run the full Exp 2 analysis and write figures + findings to output_dir."""
    if output_dir is None:
        output_dir = amgm_config.work_dir("neural_SDE") / "logs" / "exp2_fib_analysis"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    basket = dataset.basket_issue_ids
    dates = dataset.test_dates  # window-end dates, one per window
    features = dataset.features.numpy()
    prices = dataset.prices_aligned
    levels_orig = levels_original_scale(dataset)

    signed_nearest, abs_nearest = nearest_level_distances(features)
    mask = near_level_mask(features, eps)
    rets = next_day_returns(dataset)

    # Fig A + B: simultaneous visualization; prices at each window end
    w = dataset.lookback_window
    window_end_prices = prices[w - 1 : w - 1 + len(dataset)]
    plot_simultaneous_levels(dates, window_end_prices, levels_orig, basket,
                             output_dir / "figA_simultaneous_levels.png", segment_days)
    plot_distance_heatmap(dates, abs_nearest, basket, eps, output_dir / "figB_distance_heatmap.png")

    # Check (a): per-asset meaningfulness
    plot_signed_distance_histograms(signed_nearest, basket, output_dir / "check_a_signed_distance_hist.png")
    fsm_counts = fsm_events_per_asset(prices)
    plot_fsm_events(fsm_counts, basket, output_dir / "check_a_fsm_events.png")

    # Check (b): simultaneity and lead/lag
    cooc = pairwise_cooccurrence(mask)
    plot_cooccurrence_heatmap(cooc["lift"], cooc["p_near"], basket, eps, output_dir / "check_b_cooccurrence_lift.png")
    lags, xcorr = lagged_indicator_xcorr(mask, max_lag=max_lag)
    xcorr_summary = plot_lagged_xcorr(lags, xcorr, output_dir / "check_b_lagged_xcorr.png")

    # Check (c): coincidence with movements in other assets
    cond_stats = conditional_returns(mask, rets)
    top_pairs = plot_conditional_returns(
        cond_stats, basket,
        output_dir / "check_c_conditional_ks_heatmap.png",
        output_dir / "check_c_conditional_overlays.png",
    )

    write_findings(output_dir, basket, dates, mask, cooc, xcorr_summary, fsm_counts, cond_stats, top_pairs, eps)
    print(f"Exp 2 analysis written to: {output_dir}")
    return {
        "cooccurrence": cooc,
        "xcorr_summary": xcorr_summary,
        "fsm_counts": fsm_counts,
        "conditional": cond_stats,
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(name)s | %(levelname)s | %(message)s")

    parser = argparse.ArgumentParser(
        description="Experiment 2: multi-asset Fibonacci feature analysis on the aligned basket.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--trainer_cfg", default="US_Stocks_Multi",
                        help="Config module for data location; n_assets/issue_ids can be overridden below.")
    parser.add_argument("--n-assets", type=int, default=5)
    parser.add_argument("--eps", type=float, default=PAPER_BAND, help="Near-level band (normalized).")
    parser.add_argument("--segment-days", type=int, default=252, help="Days shown in the simultaneous-levels figure.")
    parser.add_argument("--max-lag", type=int, default=10)
    args = parser.parse_args()

    cfg_path = f"trainer_cfg.neural_SDE.{args.trainer_cfg}"
    cfg_module = importlib.import_module(cfg_path, package="experiments.neural_SDE")
    dset_cfg = dict(cfg_module.get_trainer_cfg()["dset_cfg"])
    dset_cfg["n_assets"] = args.n_assets

    dataset = MultiAssetNeuralSDEDataset(**dset_cfg, rng_seed=42)
    main_fib_analysis(dataset=dataset, eps=args.eps, segment_days=args.segment_days, max_lag=args.max_lag)
