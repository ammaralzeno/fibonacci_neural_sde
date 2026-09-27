"""Rebuild all eight figures and a self-contained HTML report from saved tables."""

import base64
import html
import json
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd

from .statistics import EXPERTS, PROBS, RATIOS

COLORS = ("#0072B2", "#D55E00", "#009E73")
TITLES = [
    "Expert utilization through training", "Where collapse begins",
    "Balanced, confident, or uninformative?", "Are the same inputs changing assignments?",
    "Gate probability versus Fibonacci distance", "Distance plus movement direction",
    "Do expert names match observed events?", "Concrete examples of level interactions",
]
CAPTIONS = [
    "Sample-weighted mean gates. Training averages span changing weights; validation uses the same windows each epoch. "
    "The one-third line is a reference, not a required target. Unequal use alone does not establish collapse.",
    "Faint lines are raw batch averages; bold lines are trailing 20-batch means. Vertical lines mark epoch boundaries. "
    "The horizontal axis is shuffled training order, not market time; oscillation here is not a market-regime switch rate.",
    "Exact winning ties share credit. Entropy is H(p)/log(3). High entropy and similar gates everywhere can indicate "
    "uninformative mixing despite balanced utilization. Low entropy alone is not proof of useful specialization.",
    "Identical validation windows are matched across epochs. Winner switches exclude windows with a top-two margin below "
    "0.05 in either epoch; the exclusion rate is shown. Probability change includes every window. This measures learning stability.",
    "Signed distance to the nearest rolling Fibonacci level, in historical-range units. Lines are pooled-window means; "
    "bands are pointwise 95% intervals from 1,000 stock-cluster bootstrap samples. They describe stock variation, not seed uncertainty. "
    "Bins with fewer than 30 windows or five stocks are masked. Curves are associations, not controlled or causal responses.",
    "Each cell is a mean probability on a shared 0-1 scale. Movement is the latest price change divided by the rolling range. "
    "Positive distance means above a level; movement toward zero approaches it. Grey cells have fewer than 30 windows or five stocks. "
    "Counts include unsupported cells. Distance alone cannot distinguish bouncing from breaking.",
    "The unchanged detector supplies operational event labels using fixed origin levels: tolerance 0.02, dwell 3, confirmation 10, "
    "consolidation timeout 20. Gates use rolling levels and history only, measured at exit for Bounce/Break and hover emission for Hover. "
    "Confirmation uses later observations solely to label outcomes. Hover can precede Bounce/Break; rows are stages, not exclusive regimes. "
    "The right panel subtracts the same event's entry gates. Mixture weights are not calibrated event probabilities.",
    "One lower-median-duration event per category, ordered by duration then stock, origin date, and entry index. Selection never uses "
    "gate quality. Price subtracts the origin minimum and divides by the origin range; gates use rolling windows. "
    "Event confirmation may occur after the gate measurement.",
]


def read_table(root, name):
    path = root / f"{name}.csv"
    return pd.read_csv(path, dtype={"issue_id": str}) if path.exists() else pd.DataFrame()


def missing(ax, message="Insufficient evidence"):
    ax.text(.5, .5, message, ha="center", va="center", transform=ax.transAxes, wrap=True, color="#555555")
    ax.set_xticks([])
    ax.set_yticks([])


def probability_axis(ax):
    ax.set_ylim(0, 1)
    ax.set_ylabel("Gate probability")
    ax.grid(alpha=.2)


def save(fig, root, number):
    fig.suptitle(f"{number}. {TITLES[number - 1]}", fontsize=16)
    fig.supxlabel(textwrap.fill(CAPTIONS[number - 1], width=int(fig.get_figwidth() * 15)),
                  fontsize=9, color="#444444")
    path = root / f"{number:02d}_gating.png"
    fig.savefig(path, dpi=160, facecolor="white")
    plt.close(fig)
    return path


def training_figures(root):
    epochs, batches, changes = [read_table(root, n) for n in
                                ("epoch_metrics", "training_batches", "assignment_changes")]
    figures = []
    if epochs.empty:
        for number in range(1, 5):
            fig, ax = plt.subplots(figsize=(10, 3), layout="constrained")
            missing(ax, "Training history unavailable.\nAn existing checkpoint cannot reconstruct experiment 1.")
            figures.append(save(fig, root, number))
        return figures
    epochs = epochs.sort_values("epoch")
    validation = epochs[epochs.split == "validation"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout="constrained")
    for ax, split in zip(axes, ["train", "validation"]):
        frame = epochs[epochs.split == split]
        for col, label, color in zip(PROBS, EXPERTS, COLORS):
            ax.plot(frame.epoch, frame[col], color=color, label=label, marker=".")
        ax.axhline(1 / 3, color="grey", linestyle=":", alpha=.6)
        ax.set(title=f"{split.title()} | n={int(frame.n.iloc[-1]):,} windows/epoch", xlabel="Epoch (zero-based)")
        probability_axis(ax)
        ax.legend()
    figures.append(save(fig, root, 1))

    fig, axes = plt.subplots(3, 1, figsize=(12, 7), sharex=True, layout="constrained")
    for ax, col, label, color in zip(axes, PROBS, EXPERTS, COLORS):
        ax.plot(batches.step, batches[col], color=color, alpha=.2, linewidth=.7)
        ax.plot(batches.step, batches[col].rolling(20, min_periods=1).mean(), color=color, label=label)
        for boundary in batches.groupby("epoch").step.min().iloc[1:]:
            ax.axvline(boundary, color="grey", alpha=.25, linewidth=.7)
        probability_axis(ax)
        ax.legend(loc="upper right")
    axes[-1].set_xlabel("Optimizer step (training order; not market time)")
    axes[0].set_title(f"{len(batches):,} batches; {int(batches.n.sum()):,} window observations across training")
    figures.append(save(fig, root, 2))

    fig, axes = plt.subplots(3, 1, figsize=(12, 8), layout="constrained")
    for ax, suffix, title in zip(axes[:2], ["", "_winner_share"], ["Mean utilization", "Winning share (ties split)"]):
        im = ax.imshow(validation[[p + suffix for p in PROBS]].to_numpy().T, aspect="auto", vmin=0, vmax=1, cmap="viridis")
        ax.set_yticks(range(3), EXPERTS)
        ax.set_xticks(range(len(validation)), validation.epoch.astype(int))
        ax.set(title=title, xlabel="Epoch")
        fig.colorbar(im, ax=ax, label="Probability / share", pad=.01)
    axes[2].plot(validation.epoch, validation.entropy, label="Normalized entropy", color="#6A3D9A")
    axes[2].plot(validation.epoch, validation.confidence, label="Mean maximum probability", color="#333333")
    axes[2].set(xlabel="Epoch", ylabel="Value", ylim=(0, 1))
    axes[2].legend()
    axes[2].grid(alpha=.2)
    axes[2].set_title(f"Fixed validation population: {int(validation.n.iloc[-1]):,} windows")
    figures.append(save(fig, root, 3))

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5), layout="constrained")
    for ax, col, title in zip(axes, ["switch_fraction", "mean_absolute_probability_change", "exclusion_rate"],
                               ["Winning-expert changes", "Mean absolute gate change", "Ambiguous windows excluded"]):
        if changes.empty:
            missing(ax, "At least two validation epochs required")
        else:
            ax.plot(changes.epoch, changes[col], marker="o", color="#0072B2")
            ax.set(xlabel=f"Current epoch ({int(changes.n.iloc[-1]):,} paired windows)",
                   ylabel="Fraction / probability change", ylim=(0, 1))
            ax.grid(alpha=.2)
            if col == "switch_fraction" and changes[col].isna().all():
                missing(ax, "No unambiguous paired winners")
        ax.set_title(title)
    figures.append(save(fig, root, 4))
    return figures


def distance_figure(root):
    data = read_table(root, "distance_bins")
    fig = plt.figure(figsize=(14, 12), layout="constrained")
    grid = fig.add_gridspec(4, 4, height_ratios=[1.4, .55, 1, 1])

    def draw(ax, frame, title):
        for col, label, color in zip(PROBS, EXPERTS, COLORS):
            ax.plot(frame.center, frame[col].where(frame.supported), color=color, label=label)
            ax.fill_between(frame.center, frame[col + "_low"], frame[col + "_high"], color=color, alpha=.15)
        ax.axvline(0, color="grey", linestyle=":")
        ax.set_title(title)
        probability_axis(ax)
        ax.set_xlim(frame.left.min(), frame.right.max())
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
        if not frame.supported.any():
            missing(ax, "Insufficient evidence\n(<30 windows or <5 stocks per bin)")

    pooled = data[data.level == -1]
    ax = fig.add_subplot(grid[0, :])
    draw(ax, pooled, "All nearest levels pooled | shaded bands: stock-cluster uncertainty")
    ax.legend(loc="upper right", ncol=3)
    counts = fig.add_subplot(grid[1, :], sharex=ax)
    counts.bar(pooled.center, pooled.n, width=(pooled.right - pooled.left) * .9,
               color=np.where(pooled.supported, "#999999", "#DDDDDD"))
    counts.set(xlabel="Signed distance to nearest level / rolling range", ylabel="Windows")
    for i in range(7):
        ax = fig.add_subplot(grid[2 + i // 4, i % 4])
        frame = data[data.level == i]
        draw(ax, frame, f"Nearest level {RATIOS[i]:.3f} | n={frame.n.sum():,}")
        ax.set_xlabel("Signed distance")
    ax = fig.add_subplot(grid[3, 3])
    missing(ax, "Gaps are unsupported bins.\nNearest-level changes can cause\napparent distance discontinuities.\n\nEndpoint levels have\none-sided support.")
    ax.set_axis_off()
    return save(fig, root, 5)


def movement_figure(root):
    data = read_table(root, "movement_bins")
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), layout="constrained")
    extent = [data.distance_left.min(), data.distance_right.max(), data.movement_left.min(), data.movement_right.max()]
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("#DDDDDD")
    for ax, col, name in zip(axes[0], PROBS, EXPERTS):
        values = data[col].where(data.supported).to_numpy().reshape(20, 20).T
        im = ax.imshow(values, origin="lower", extent=extent, aspect="auto", vmin=0, vmax=1, cmap=cmap)
        ax.axhline(0, color="white", linewidth=.6)
        ax.axvline(0, color="white", linewidth=.6)
        ax.set(title=name, xlabel="Signed distance / range", ylabel="Recent change / range")
    fig.colorbar(im, ax=list(axes[0]), label="Mean gate probability", shrink=.8)
    for ax, col, title in zip(axes[1, :2], ["n", "stocks"], ["Window count", "Distinct stock count"]):
        im = ax.imshow(data[col].to_numpy().reshape(20, 20).T, origin="lower", extent=extent, aspect="auto", cmap="Greys")
        ax.set(title=title, xlabel="Signed distance / range", ylabel="Recent change / range")
        fig.colorbar(im, ax=ax, shrink=.8)
    missing(axes[1, 2], "Above + falling: approaching\nBelow + rising: approaching\n\nAbove + rising: moving away\nBelow + falling: moving away\n\nGrey probability cells:\ninsufficient evidence")
    axes[1, 2].set_axis_off()
    return save(fig, root, 6)


def event_figure(root):
    data = read_table(root, "event_summary")
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), layout="constrained")
    for ax, col, title, limit, cmap_name in zip(axes, ["probability", "change"],
            ["Gate at observed event", "Change from same event's entry"], [(0, 1), (-1, 1)], ["viridis", "RdBu_r"]):
        matrix = np.full((3, 3), np.nan)
        labels = []
        for i, event in enumerate(EXPERTS):
            group = data[data.event == event].set_index("expert").reindex(PROBS)
            labels.append(f"{event}\nn={int(group.n.iloc[0])}, stocks={int(group.stocks.iloc[0])}")
            matrix[i] = group[col].where(group.supported).to_numpy()
        cmap = plt.get_cmap(cmap_name).copy()
        cmap.set_bad("#DDDDDD")
        im = ax.imshow(matrix, cmap=cmap, vmin=limit[0], vmax=limit[1], aspect="auto")
        ax.set_xticks(range(3), EXPERTS)
        ax.set_yticks(range(3), labels)
        ax.set(title=title, xlabel="Expert", ylabel="Detected event stage")
        for i in range(3):
            for j in range(3):
                val = matrix[i, j]
                text = (f"{val:+.2e}" if col == "change" else f"{val:.3f}") if np.isfinite(val) else "Insufficient\nevidence"
                ax.text(j, i, text, ha="center", va="center", fontsize=10,
                        color="white" if np.isfinite(val) and col == "probability" and val < .4 else "black")
        fig.colorbar(im, ax=ax, label="Probability" if col == "probability" else "Probability difference", shrink=.8)
    return save(fig, root, 7)


def example_figure(root):
    events, samples = read_table(root, "events"), read_table(root, "real_gates")
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex="col", layout="constrained")
    marker_styles = [("entry_index", "Entry", ":"), ("hover_index", "Hover", "--"),
                     ("exit_index", "Exit", "-."), ("confirmation_end_index", "Confirmed", "-")]
    for column, name in enumerate(EXPERTS):
        selected = events[(events.event == name) & events.selected_example.astype(bool)]
        if selected.empty:
            for ax in axes[:, column]:
                missing(ax, f"{name}: insufficient evidence\nNo observed event")
            continue
        e = selected.iloc[0]
        path = samples[samples.segment_id == e.segment_id].sort_values("step")
        top, bottom = axes[:, column]
        top.plot(path.step, path.fixed_price, color="#333333", label="Real price")
        top.axhline(e.fib_level, color="#B58900", label=f"Fixed level {e.fib_level:.3f}")
        top.axhspan(e.fib_level - .02, e.fib_level + .02, color="#B58900", alpha=.15)
        top.set(title=f"{name} | {e.issue_id}\nOrigin {e.origin_date}; duration {int(e.duration)} steps",
                ylabel="Price (origin-normalized)")
        for key, label, linestyle in marker_styles:
            if pd.notna(e[key]):
                for ax in [top, bottom]:
                    ax.axvline(e[key], color="#777777", linestyle=linestyle, linewidth=.9,
                               label=label if ax is top else None)
        for col, label, color in zip(PROBS, EXPERTS, COLORS):
            bottom.plot(path.step, path[col], color=color, label=label)
        bottom.axvline(e.gate_index, color="#000000", linewidth=1.4, alpha=.5)
        probability_axis(bottom)
        bottom.set_xlabel("Trading steps from origin")
        top.legend(fontsize=8, loc="best")
        bottom.legend(fontsize=8, loc="best")
    return save(fig, root, 8)


def render_report(output_dir):
    root = Path(output_dir)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    with plt.rc_context({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False}):
        figures = training_figures(root)
        figures += [distance_figure(root), movement_figure(root), event_figure(root), example_figure(root)]
    counts = manifest["evaluation_counts"]
    content = ["<!doctype html><html lang='en'><meta charset='utf-8'><title>MoE gating diagnostics</title>",
               "<style>body{font:17px/1.55 system-ui,sans-serif;max-width:1200px;margin:40px auto;padding:0 24px;color:#222}"
               "img{width:100%;height:auto}figure{margin:36px 0}figcaption{color:#444}table{border-collapse:collapse;font-size:14px}"
               "td,th{padding:7px;border-bottom:1px solid #ddd}code{overflow-wrap:anywhere}h1,h2{line-height:1.2}</style>",
               "<h1>Do the gates mean what their names suggest?</h1>",
               "<p>Experiments 1 and 2: observational diagnostics of the unchanged MoE. "
               "Balanced averages do not establish useful specialization, and unequal averages alone do not establish collapse.</p>",
               f"<p><b>Training history:</b> {html.escape(manifest['experiment1_status'])}. "
               f"<b>Checkpoint epoch:</b> {manifest.get('checkpoint_epoch', 'unknown')} (zero-based). "
               f"<b>Evaluation:</b> {counts['windows']:,} windows, {counts['stocks']} stocks, "
               f"{counts['segments']} segments in 2018–2020.</p>",
               f"<p><b>Detector diagnostics:</b> {counts['timeouts']} timeouts; {counts['incomplete']} unresolved "
               "interactions at segment ends. Unresolved counts use constant extension only to identify truncated lifecycles; "
               "extended observations never enter model inputs, event statistics, or plots. "
               f"{counts['exclusions']} excluded stocks/segments are listed in exclusions.csv.</p>",
               "<p>Training validation contains randomly split overlapping windows and is used here to monitor training, "
               "not to claim independent forecasting performance. Later-period evaluation preserves within-stock dependence "
               "through stock-cluster bootstrap intervals. One training seed cannot establish reproducibility across seeds.</p>"]
    epochs = read_table(root, "epoch_metrics")
    if not epochs.empty:
        last = epochs[epochs.split == "validation"].sort_values("epoch").iloc[-1]
        content.append("<p><b>Final validation snapshot:</b> " + ", ".join(
            f"{name} mean={last[col]:.3f}, SD={np.sqrt(last[col + '_variance']):.3f}"
            for col, name in zip(PROBS, EXPERTS)) + f"; normalized entropy={last.entropy:.3f}. "
            "This is the last epoch; offline event analysis uses the best checkpoint identified above.</p>")
    event_metrics = read_table(root, "event_summary")
    if event_metrics.supported.all():
        matrix = event_metrics.pivot(index="event", columns="expert", values="probability").reindex(EXPERTS)
        winners = matrix.idxmax(axis=1)
        if winners.nunique() == 1:
            winner = EXPERTS[PROBS.index(winners.iloc[0])]
            content.append(f"<p><b>Event alignment:</b> {winner} has the highest mean gate in all three observed "
                           "event stages. The event names therefore do not correspond to three different dominant experts in this run. "
                           "This is descriptive evidence, not a test of calibrated probabilities.</p>")
        content.append(f"<p>The largest difference between event-stage mean probabilities for any expert is "
                       f"{(matrix.max() - matrix.min()).max():.4f}. The largest absolute mean change from entry "
                       f"is {event_metrics.change.abs().max():.4f}. See the stock counts and stage definitions below.</p>")
    for i, path in enumerate(figures):
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        content.append(f"<figure><h2>{i + 1}. {TITLES[i]}</h2><img alt='{html.escape(TITLES[i])}' "
                       f"src='data:image/png;base64,{encoded}'><figcaption>{CAPTIONS[i]}</figcaption></figure>")
    content.append("<h2>Event summary</h2>" + read_table(root, "event_summary").to_html(index=False, float_format=lambda x: f"{x:.4f}"))
    content.append("<h2>Reproducibility</h2><p>All figures are rebuilt from the CSV tables beside this report. "
                   "Validation NPZ files retain per-window epoch gates. real_windows.npz retains offline gates and all seven distances. "
                   "The manifest records configuration, stock selection, and checkpoint SHA-256.</p><pre style='white-space:pre-wrap'>" +
                   html.escape(json.dumps(manifest, indent=2)) + "</pre></html>")
    (root / "report.html").write_text("\n".join(content), encoding="utf-8")
