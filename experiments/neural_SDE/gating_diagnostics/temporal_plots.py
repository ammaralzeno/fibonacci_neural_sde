"""Five temporal figures and a self-contained report, reconstructed from CSVs."""

import base64
import html
import json
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.style.use("default")

import numpy as np
import pandas as pd

from .statistics import EXPERTS, PROBS

EXPERT_COLORS = ("#0072B2", "#D55E00", "#009E73")
SOURCE_COLORS = {"real": "#333333", "synthetic": "#8E44AD"}
TITLES = [
    "Paired real and synthetic trajectories",
    "Expert switches per trajectory",
    "Observed expert dwell lengths",
    "Expert-state transition probabilities",
    "Soft gate changes",
]
CAPTIONS = [
    "Origins are evenly spaced in stock/date order; synthetic path ID 0 is always selected, irrespective of gates. "
    "Each pair shares its initial 252-observation history. Steps 0-100 give 101 gates and 100 transitions. Vertical lines mark winner changes.",
    "Each origin has equal weight; valid synthetic paths share their origin's weight. A switch is any argmax change, including ambiguous winners. "
    "Zero switches can reflect expert dominance rather than meaningful stability.",
    "Duration counts observed consecutive labels. Solid lines: all observed runs; dashed: fully observed interior contributions. "
    "Their gap is boundary-censored mass. First/last runs have unknown full durations; 101 is a window-length lower bound, not a completed regime duration.",
    "Rows condition on the current winning expert; columns identify the next expert. Counts are origin-weighted before normalization. "
    "Grey rows are absent or have fewer than 30 raw transitions or five stocks. Missing states are not zero transition probabilities.",
    "Soft change is mean_k |pi_k(t)-pi_k(t-1)|. Left: timestep changes; right: trajectory means. Both are origin-weighted. "
    "Bins span the observed range without clipping tails; distributions measure probability movement even when the winning expert never changes.",
]


def read(root, name):
    try:
        return pd.read_csv(root / f"{name}.csv", dtype={"issue_id": str})
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def no_data(ax, message="Insufficient evidence"):
    ax.text(
        0.5, 0.5, message, ha="center", va="center", transform=ax.transAxes, fontsize=10
    )
    ax.set_xticks([])
    ax.set_yticks([])


def save(fig, root, number, counts):
    fig.suptitle(f"{number}. {TITLES[number - 1]}", fontsize=16)
    caption = CAPTIONS[number - 1] + " " + counts
    fig.supxlabel(
        textwrap.fill(caption, width=int(fig.get_figwidth() * 14)),
        fontsize=9,
        color="#444444",
    )
    target = root / f"{number:02d}_temporal.png"
    fig.savefig(target, dpi=160, facecolor="white")
    plt.close(fig)
    return target


def examples(root, manifest, counts):
    frame = read(root, "example_steps")
    ids = manifest["example_segments"]
    fig, axes = plt.subplots(
        4,
        len(ids),
        figsize=(5 * len(ids), 11),
        squeeze=False,
        sharex="col",
        layout="constrained",
    )
    for col, sid in enumerate(ids):
        pair = frame[frame.segment_id == sid]
        for offset, source in [(0, "real"), (2, "synthetic")]:
            group = pair[pair.source == source].sort_values("step")
            top, bottom = axes[offset, col], axes[offset + 1, col]
            if group.empty:
                no_data(top)
                no_data(bottom)
                continue
            first = group.iloc[0]
            top.set_title(
                f"{source.title()} | {first.issue_id}\n{first.origin_date}"
                + (" | path 0" if source == "synthetic" else "")
            )
            if not group.valid_path.all():
                no_data(top, "Selected path invalid\nNot replaced with another path")
                no_data(bottom, "Excluded from primary comparisons")
                continue
            top.plot(group.step, group.price, color=SOURCE_COLORS[source])
            top.set_ylabel("Adjusted price (source units)")
            for p, name, color in zip(PROBS, EXPERTS, EXPERT_COLORS):
                bottom.plot(group.step, group[p], color=color, label=name)
            switch_steps = group.step.iloc[
                np.flatnonzero(np.diff(group.state.to_numpy()) != 0) + 1
            ]
            for step in switch_steps:
                top.axvline(step, color="grey", alpha=0.3, linewidth=0.6)
                bottom.axvline(step, color="grey", alpha=0.3, linewidth=0.6)
            bottom.set(
                ylim=(0, 1),
                ylabel="Gate probability",
                xlabel="Trading steps from origin",
            )
            bottom.legend(fontsize=8, loc="best")
            bottom.grid(alpha=0.2)
        if pair.valid_path.all():
            low, high = pair.price.min(), pair.price.max()
            pad = max((high - low) * 0.05, 1e-5)
            for offset in [0, 2]:
                axes[offset, col].set_ylim(low - pad, high + pad)
    return save(fig, root, 1, counts)


def switches(root, counts):
    distributions, summary = read(root, "distributions"), read(root, "comparisons")
    fig, ax = plt.subplots(figsize=(11, 5), layout="constrained")
    for source, color in SOURCE_COLORS.items():
        g = distributions[
            (distributions.metric == "switch_count") & (distributions.source == source)
        ]
        if summary.empty or not np.isfinite(g.probability).any():
            continue
        s = summary[
            (summary.source == source) & (summary.metric == "switch_count")
        ].iloc[0]
        zero = summary[
            (summary.source == source) & (summary.metric == "zero_switch")
        ].iloc[0]["mean"]
        ax.step(
            g.center,
            g.probability,
            where="mid",
            color=color,
            label=f"{source.title()}: mean {s['mean']:.3f}, median {s['median']:.0f}, zero {zero:.1%}",
        )
        ax.scatter(g.center, g.probability, color=color, s=8)
    if summary.empty:
        no_data(ax, "No valid paired origins")
    else:
        ax.legend()
    ax.set(
        xlabel="Switches per 100 transitions",
        ylabel="Weighted probability",
        ylim=(0, 1.05),
        xlim=(-1, 101),
    )
    ax.grid(alpha=0.2)
    return save(fig, root, 2, counts)


def dwell_figure(root, counts):
    d = read(root, "dwell_distribution")
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), layout="constrained")
    for ax, name, color in zip(axes, EXPERTS, EXPERT_COLORS):
        group = d[d.expert == name]
        ax.set_title(name, color=color)
        if not np.isfinite(group.probability).any():
            no_data(ax, f"No observed {name} runs\nPersistence cannot be estimated")
            continue
        for source, source_color in SOURCE_COLORS.items():
            g = group[group.source == source]
            all_runs = g.groupby("duration").probability.sum(min_count=1)
            interior = g[g.interior].set_index("duration").probability
            ax.plot(
                all_runs.index,
                all_runs,
                color=source_color,
                marker=".",
                label=f"{source.title()}: all",
            )
            ax.plot(
                interior.index,
                interior,
                color=source_color,
                linestyle="--",
                label=f"{source.title()}: interior",
            )
        ax.set(
            xlabel="Observed dwell length (timesteps)",
            ylabel="Weighted probability",
            ylim=(0, 1.05),
            xlim=(0, 103),
        )
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)
    return save(fig, root, 3, counts)


def transition_figure(root, counts):
    table = read(root, "transition_probabilities")
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.5), layout="constrained")
    for ax, source in zip(axes, ["real", "synthetic", "synthetic-minus-real"]):
        difference = source == "synthetic-minus-real"
        ax.set_title(source.replace("-", " ").title())
        if table.empty:
            no_data(ax, "No valid paired origins")
            continue
        g = table[table.source == source]
        matrix = (
            g.pivot(index="from_expert", columns="to_expert", values="probability")
            .reindex(index=EXPERTS, columns=EXPERTS)
            .to_numpy()
        )
        supported = (
            g.pivot(index="from_expert", columns="to_expert", values="supported")
            .reindex(index=EXPERTS, columns=EXPERTS)
            .to_numpy()
            .astype(bool)
        )
        cmap = plt.get_cmap("RdBu_r" if difference else "viridis").copy()
        cmap.set_bad("#dddddd")
        im = ax.imshow(
            np.where(supported, matrix, np.nan),
            cmap=cmap,
            vmin=-1 if difference else 0,
            vmax=1,
        )
        ax.set_xticks(range(3), EXPERTS)
        labels = []
        for i, name in enumerate(EXPERTS):
            r = g[g.from_expert == name].iloc[0]
            labels.append(
                name
                if difference
                else f"{name}\nn={r.raw_source_count:,.0f}; stocks={r.stocks:.0f}"
            )
            for j in range(3):
                value = matrix[i, j]
                label = (
                    "Undefined"
                    if not np.isfinite(value)
                    else f"{value:+.4f}"
                    if difference
                    else f"{value:.4f}"
                )
                if np.isfinite(value) and not supported[i, j]:
                    label += "\nLow coverage"
                ax.text(
                    j,
                    i,
                    label,
                    ha="center",
                    va="center",
                    fontsize=9,
                    color="white"
                    if supported[i, j] and not difference and value < 0.5
                    else "black",
                )
        ax.set_yticks(range(3), labels)
        ax.set(xlabel="Next expert", ylabel="Current expert")
        fig.colorbar(
            im, ax=ax, shrink=0.7, label="Difference" if difference else "Probability"
        )
    return save(fig, root, 4, counts)


def soft_figure(root, counts):
    table = read(root, "distributions")
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), layout="constrained")
    for ax, metric, title in zip(
        axes,
        ["step_soft_change", "soft_change_mean"],
        ["Individual timestep changes", "Mean change per trajectory"],
    ):
        data = table[table.metric == metric]
        for source, color in SOURCE_COLORS.items():
            g = data[data.source == source]
            if np.isfinite(g.probability).any():
                ax.stairs(
                    g.probability,
                    np.r_[g.left, g.right.iloc[-1]],
                    color=color,
                    label=source.title(),
                )
        if not np.isfinite(data.probability).any():
            no_data(ax, "No valid paired origins")
        else:
            ax.legend()
        ax.set(
            title=title,
            xlabel="Mean absolute probability change",
            ylabel="Weighted probability per bin",
            ylim=(0, 1.05),
        )
        ax.grid(alpha=0.2)
    return save(fig, root, 5, counts)


def render_temporal_report(output_dir):
    root = Path(output_dir)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    coverage = read(root, "coverage")
    paired = coverage[coverage.paired]
    count_text = (
        f"Paired cohort: {len(paired)} origins, {paired.issue_id.nunique()} stocks, "
        f"{int(paired.valid_synthetic.sum()):,} valid synthetic paths."
    )
    with plt.rc_context(
        {"font.size": 10, "axes.spines.top": False, "axes.spines.right": False}
    ):
        figures = [
            examples(root, manifest, count_text),
            switches(root, count_text),
            dwell_figure(root, count_text),
            transition_figure(root, count_text),
            soft_figure(root, count_text),
        ]
    summary, transition = (
        read(root, "comparisons"),
        read(root, "transition_probabilities"),
    )
    content = [
        "<!doctype html><html lang='en'><meta charset='utf-8'><title>Temporal gating diagnostics</title>",
        "<style>body{font:17px/1.55 system-ui,sans-serif;max-width:1300px;margin:40px auto;padding:0 24px;color:#222}"
        "img{width:100%}figure{margin:36px 0}figcaption{color:#444}table{border-collapse:collapse;font-size:14px}"
        "td,th{padding:7px;border-bottom:1px solid #ddd}.table{overflow-x:auto}pre{white-space:pre-wrap}</style>",
        "<h1>Do synthetic paths reproduce temporal gating behavior?</h1>",
        f"<p>{count_text} Checkpoint epoch {manifest['checkpoint_epoch']} (zero-based); seed {manifest['seed']}.</p>",
        f"<p>Generated {int(coverage.generated.sum()):,} synthetic paths; {int(coverage.invalid_synthetic.sum()):,} were invalid. "
        f"{int((~coverage.paired).sum())} origins were excluded from paired comparisons. No paths were repaired or replaced. "
        "Synthetic summaries are conditional on complete, positive, finite paths; excluded paths are also relevant to generator realism.</p>",
        "<p><b>Interpretation:</b> Bounce, Break, and Hover are expert-state labels, not proven market regimes. "
        "Experiments 1-2 showed weak semantic alignment and underused Bounce. Few switches can simply reflect dominant Hover weights. "
        "Matching constant winners does not establish meaningful regime persistence or realistic synthetic prices.</p>",
        "<p>Means weight origins equally; synthetic paths share their origin's weight. Intervals are pointwise 95% paired stock-cluster "
        "bootstrap intervals from 1,000 resamples. They describe sampled-stock variation, not training-seed uncertainty. "
        "A narrow or zero-width interval for constant gates is not evidence of calibrated regime recognition.</p>",
    ]
    if not summary.empty:
        content.append("<h2>Measured behavior</h2><ul>")
        for source in ["real", "synthetic"]:
            s = summary[summary.source == source].set_index("metric")
            content.append(
                f"<li><b>{source.title()}:</b> mean switches {s.loc['switch_count', 'mean']:.4f} per 100 transitions; "
                f"zero-switch share {s.loc['zero_switch', 'mean']:.1%}; mean soft change {s.loc['soft_change_mean', 'mean']:.6f}; "
                f"ambiguous winners {s.loc['ambiguous_fraction', 'mean']:.1%}. "
                + ", ".join(
                    f"{name} mean weight {s.loc[p, 'mean']:.3f}, winning share {s.loc[p + '_winner_share', 'mean']:.1%}"
                    for name, p in zip(EXPERTS, PROBS)
                )
                + ".</li>"
            )
        content.append(
            "</ul><p>Compare switching and soft changes together. Similar dominant-expert occupancy may conceal differences in "
            "probability movement. Absent expert states prevent estimating their persistence or exit behavior.</p>"
        )
        zero = summary[
            (summary.source.isin(["real", "synthetic"]))
            & (summary.metric == "zero_switch")
        ]
        if zero["mean"].eq(1).all():
            content.append(
                "<p><b>Hard-state result:</b> every included real and synthetic trajectory has zero switches. "
                "All observed dwell runs span the full 101-observation window and are boundary-censored. "
                "This matches persistence within these windows, but provides no completed dwell times or evidence "
                "of three distinct temporal expert regimes.</p>"
            )
        delta = summary[
            (summary.source == "synthetic-minus-real")
            & (summary.metric == "soft_change_mean")
        ].iloc[0]
        content.append(
            f"<p><b>Soft-state comparison:</b> synthetic-minus-real mean absolute gate change is "
            f"{delta['mean']:+.6f}, with a pointwise 95% paired stock-bootstrap interval "
            f"[{delta.low:+.6f}, {delta.high:+.6f}]. "
            "Matching winner labels alone does not establish matching probability dynamics.</p>"
        )
        real_change = summary[
            (summary.source == "real") & (summary.metric == "soft_change_mean")
        ]["mean"].iloc[0]
        if real_change > 0 and delta.low > 0:
            content.append(
                f"<p>Mean synthetic probability movement is {100 * delta['mean'] / real_change:.1f}% greater than real "
                "probability movement in this cohort. The paired interval is above zero, providing evidence of a difference "
                "in average soft-gate movement, despite matching hard-state persistence. This does not measure downstream classifier utility.</p>"
            )
    else:
        content.append(
            "<p><b>Insufficient evidence:</b> no valid paired origins remain.</p>"
        )
    for i, path in enumerate(figures):
        content.append(
            f"<figure><h2>{i + 1}. {TITLES[i]}</h2><img alt='{TITLES[i]}' src='data:image/png;base64,"
            + base64.b64encode(path.read_bytes()).decode()
            + f"'><figcaption>{CAPTIONS[i]} {count_text}</figcaption></figure>"
        )
    for title, data in [
        ("Summary comparisons and confidence intervals", summary),
        ("Transition estimates and intervals", transition),
        ("Observed versus interior dwell summaries", read(root, "dwell_summary")),
    ]:
        content.append(
            f"<h2>{title}</h2><div class='table'>"
            + data.to_html(index=False, float_format=lambda x: f"{x:.6g}")
            + "</div>"
        )
    content.append(
        "<h2>Reproduce and audit</h2><p>Figures read CSV tables only. steps.csv.gz contains all recorded timesteps; "
        "origins/ contains compressed per-origin arrays, seeds, and validity flags. Dates on synthetic paths are matched "
        "trading-calendar positions, not actual future observations. No cross-segment transitions are counted.</p><pre>"
        + html.escape(json.dumps(manifest, indent=2))
        + "</pre></html>"
    )
    (root / "report.html").write_text("\n".join(content), encoding="utf-8")
