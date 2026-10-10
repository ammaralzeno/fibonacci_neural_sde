"""Experiment 3: compare ablation results across baskets of different dependency strength.

Aggregates the per-epoch metrics of every exp3 run (one ablation family per basket,
multiple seeds) and writes the cross-basket deliverables to the output dir:
  - exp3_basket_comparison.csv: best validation NLL (mean/std over seeds) per basket x config
  - exp3_basket_scaling.png: NLL gain vs dependency strength with seed error bars
  - exp3_basket_findings.md: slide-ready summary
"""

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from amgm import config as amgm_config
from experiments.neural_SDE.compare_ablations_multi import EXP3_CONFIGS, config_of, load_val_curves

BASKET_SUFFIX = {"same-sector": "", "concentrated": "_conc", "diversified": "_div"}
SEEDS = (42, 43, 44, 45, 46, 47)  # seed-42 runs keep their original names

# Period comparison on the Exp 1 baskets (correlation is stronger in 2018+)
PERIOD_SUFFIX = {"2015-2017": "", "2018-2020": "_late", "2014-2020": "_all"}
PERIOD_BASKETS = {"concentrated": "_conc", "diversified": "_div"}


def run_name(config, basket, seed):
    name = config + BASKET_SUFFIX[basket]
    return name if seed == SEEDS[0] else f"{name}_s{seed}"


def _read_corr_init(log_root, name):
    """Training increment correlation the model was initialized with (from its hparams)."""
    # BaseLoader: hparams.yaml contains python/name tags that safe_load rejects
    hparams = yaml.load((Path(log_root) / name / "hparams.yaml").read_text(), Loader=yaml.BaseLoader)
    return np.asarray(hparams["model_cfg"]["corr_init"], dtype=float)


def dependency_strength(corr):
    """Mean off-diagonal correlation of the basket's training increments."""
    return float(corr[~np.eye(len(corr), dtype=bool)].mean())


def correlation_ceiling(corr):
    """-0.5 * log det R: the NLL gain scale the correlation structure can explain."""
    sign, logdet = np.linalg.slogdet(corr)
    return float(-0.5 * logdet)


def build_basket_table(log_root, baskets=None, seeds=SEEDS):
    """One row per basket x configuration: best validation NLL (mean/std over seeds)
    and delta vs the basket's independent run of the same seed."""
    baskets = baskets or list(BASKET_SUFFIX)
    configs = list(EXP3_CONFIGS)
    all_runs = [run_name(c, b, s) for b in baskets for c in configs for s in seeds]
    curves = load_val_curves(log_root, all_runs)

    rows = []
    for basket in baskets:
        corr = _read_corr_init(log_root, run_name(configs[0], basket, seeds[0]))
        strength, ceiling = dependency_strength(corr), correlation_ceiling(corr)
        best = {
            (c, s): float(curves[run_name(c, basket, s)]["val/loss_sde"].min())
            for c in configs for s in seeds
        }
        for c in configs:
            _, _, label = config_of(c)
            nlls = np.array([best[(c, s)] for s in seeds])
            deltas = np.array([best[(c, s)] - best[(configs[0], s)] for s in seeds])
            rows.append({
                "basket": basket,
                "mean_pairwise_corr": strength,
                "corr_ceiling": ceiling,
                "config": c,
                "label": label,
                "nll_mean": nlls.mean(),
                "nll_std": nlls.std(),
                "delta_mean": deltas.mean(),
                "delta_std": deltas.std(),
            })
    return pd.DataFrame(rows)


def plot_basket_scaling(table, output_file):
    """Validation NLL gain vs basket dependency strength, one line per configuration,
    error bars = std over seeds."""
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    for label, grp in table.groupby("label", sort=False):
        if (grp["delta_mean"] == 0).all() and (grp["delta_std"] == 0).all():
            continue  # the independent baseline is the zero line
        grp = grp.sort_values("mean_pairwise_corr")
        ax.errorbar(
            grp["mean_pairwise_corr"], grp["delta_mean"], yerr=grp["delta_std"],
            marker="o", capsize=4, linewidth=1.4, label=label,
        )
    ax.axhline(0, color="black", linewidth=0.8, label="Independent baseline")
    ax.set_xlabel("basket dependency strength (mean pairwise increment correlation)")
    ax.set_ylabel("delta best validation NLL vs independent")
    ax.set_title("Experiment 3: joint-modeling value vs cross-asset dependency strength", fontsize=11)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_file, dpi=150, bbox_inches="tight")
    plt.close(fig)


def write_basket_findings(output_dir, table):
    """One-screen cross-basket Exp 3 findings."""
    lines = [
        "# Exp 3 basket comparison",
        "",
        f"Delta best val NLL vs independent (mean +/- std, {len(SEEDS)} seeds; negative = joint helps).",
        "Ceiling = -0.5*logdet(R): the NLL gain the basket's correlation structure can explain.",
        "",
        "| Basket | mean corr | ceiling | Correlation only | Context only | Full joint |",
        "|---|---|---|---|---|---|",
    ]
    for basket, grp in table.groupby("basket", sort=False):
        r0 = grp.iloc[0]
        cells = []
        for cfg in ("exp3_corr_only", "exp3_context_only", "exp3_full"):
            r = grp[grp["config"] == cfg].iloc[0]
            cells.append(f"{r['delta_mean']:+.2f} +/- {r['delta_std']:.2f}")
        lines.append(f"| {basket} | {r0['mean_pairwise_corr']:.2f} | {r0['corr_ceiling']:.2f} | " + " | ".join(cells) + " |")
    lines += [
        "",
        "Gain tracks the ceiling: decisive on same-sector; within noise on concentrated/diversified (2015-2017).",
        "Figure: `exp3_basket_scaling.png`. Data: `exp3_basket_comparison.csv`.",
    ]
    (Path(output_dir) / "exp3_basket_findings.md").write_text("\n".join(lines) + "\n")


def main_compare_baskets(log_root=None, output_dir=None):
    """Build the cross-basket Exp 3 deliverables from the finished ablation runs."""
    if log_root is None:
        log_root = amgm_config.work_dir("neural_SDE") / "logs" / "train_neural_SDE_multi"
    if output_dir is None:
        output_dir = amgm_config.work_dir("neural_SDE") / "logs" / "exp3_ablations"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    table = build_basket_table(log_root)
    table.to_csv(output_dir / "exp3_basket_comparison.csv", index=False)
    plot_basket_scaling(table, output_dir / "exp3_basket_scaling.png")
    write_basket_findings(output_dir, table)
    print(table.to_string(index=False))
    print(f"Cross-basket Exp 3 deliverables written to: {output_dir}")
    return table


# ---------------------------------------------------------------------------
# Period comparison (same baskets, different training periods)
# ---------------------------------------------------------------------------


def _period_run_name(config, basket, period, seed):
    name = config + PERIOD_BASKETS[basket] + PERIOD_SUFFIX[period]
    return name if seed == SEEDS[0] else f"{name}_s{seed}"


def build_period_table(log_root, seeds=SEEDS):
    """One row per basket x period x configuration: best validation NLL (mean/std over
    seeds) and delta vs the independent run of the same basket, period, and seed."""
    configs = list(EXP3_CONFIGS)
    baskets, periods = list(PERIOD_BASKETS), list(PERIOD_SUFFIX)
    all_runs = [_period_run_name(c, b, p, s) for b in baskets for p in periods for c in configs for s in seeds]
    curves = load_val_curves(log_root, all_runs)

    rows = []
    for basket in baskets:
        for period in periods:
            corr = _read_corr_init(log_root, _period_run_name(configs[0], basket, period, seeds[0]))
            best = {
                (c, s): float(curves[_period_run_name(c, basket, period, s)]["val/loss_sde"].min())
                for c in configs for s in seeds
            }
            for c in configs:
                _, _, label = config_of(c)
                nlls = np.array([best[(c, s)] for s in seeds])
                deltas = np.array([best[(c, s)] - best[(configs[0], s)] for s in seeds])
                rows.append({
                    "basket": basket,
                    "period": period,
                    "mean_pairwise_corr": dependency_strength(corr),
                    "corr_ceiling": correlation_ceiling(corr),
                    "config": c,
                    "label": label,
                    "nll_mean": nlls.mean(),
                    "nll_std": nlls.std(),
                    "delta_mean": deltas.mean(),
                    "delta_std": deltas.std(),
                })
    return pd.DataFrame(rows)


def plot_period_scaling(table, output_file):
    """Correlation-channel NLL gain per basket across training periods (error bars = seed std)."""
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    corr_only = table[table["config"] == "exp3_corr_only"]
    for basket, grp in corr_only.groupby("basket"):
        grp = grp.set_index("period").loc[list(PERIOD_SUFFIX)].reset_index()
        ax.errorbar(
            grp["period"], grp["delta_mean"], yerr=grp["delta_std"],
            marker="o", capsize=4, linewidth=1.4, label=basket,
        )
    ax.axhline(0, color="black", linewidth=0.8, label="Independent baseline")
    ax.set_ylabel("delta best validation NLL vs independent")
    ax.set_title("Experiment 3: correlation-channel gain vs training period", fontsize=11)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_file, dpi=150, bbox_inches="tight")
    plt.close(fig)


def write_period_findings(output_dir, table):
    """One-screen Exp 3 period findings."""
    lines = [
        "# Exp 3 period comparison",
        "",
        f"Delta best val NLL vs independent (mean +/- std, {len(SEEDS)} seeds).",
        "Rolling correlation is ~50-65% stronger in 2018-2020 than 2015-2017 (Exp 1).",
        "",
        "| Basket | Period | mean corr | ceiling | Correlation only | Full joint |",
        "|---|---|---|---|---|---|",
    ]
    for period in PERIOD_SUFFIX:
        for basket in PERIOD_BASKETS:
            sub = table[(table["basket"] == basket) & (table["period"] == period)]
            co = sub[sub["config"] == "exp3_corr_only"].iloc[0]
            fj = sub[sub["config"] == "exp3_full"].iloc[0]
            lines.append(
                f"| {basket} | {period} | {co['mean_pairwise_corr']:.2f} | {co['corr_ceiling']:.2f} | "
                f"{co['delta_mean']:+.3f} +/- {co['delta_std']:.3f} | {fj['delta_mean']:+.3f} +/- {fj['delta_std']:.3f} |"
            )
    lines += [
        "",
        "Figure: `exp3_period_scaling.png`. Data: `exp3_period_comparison.csv`.",
    ]
    (Path(output_dir) / "exp3_period_findings.md").write_text("\n".join(lines) + "\n")


def main_compare_periods(log_root=None, output_dir=None):
    """Build the Exp 3 period-comparison deliverables from the finished runs."""
    if log_root is None:
        log_root = amgm_config.work_dir("neural_SDE") / "logs" / "train_neural_SDE_multi"
    if output_dir is None:
        output_dir = amgm_config.work_dir("neural_SDE") / "logs" / "exp3_ablations"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    table = build_period_table(log_root)
    table.to_csv(output_dir / "exp3_period_comparison.csv", index=False)
    plot_period_scaling(table, output_dir / "exp3_period_scaling.png")
    write_period_findings(output_dir, table)
    print(table.to_string(index=False))
    print(f"Exp 3 period deliverables written to: {output_dir}")
    return table


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(name)s | %(levelname)s | %(message)s")
    import argparse

    parser = argparse.ArgumentParser(description="Exp 3 comparisons.")
    parser.add_argument("--periods", action="store_true",
                        help="Compare training periods on the concentrated/diversified baskets.")
    args = parser.parse_args()
    if args.periods:
        main_compare_periods()
    else:
        main_compare_baskets()
