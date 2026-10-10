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
SEEDS = (42, 43, 44)  # seed-42 runs keep their original names


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
    """Slide-ready cross-basket Exp 3 findings."""
    lines = [
        "# Experiment 3 — basket comparison: joint-modeling value vs dependency strength",
        "",
        f"Same ablation protocol on three baskets of increasing cross-asset dependency,",
        f"seeds {SEEDS} (mean +/- std). Validation NLL: lower is better; delta is vs the",
        "basket's own independent baseline of the same seed.",
        "",
        "The correlation ceiling -0.5*logdet(R) upper-bounds the NLL gain learnable from",
        "the basket's correlation structure — it ranks the baskets before any training.",
        "",
        "| Basket | mean corr | ceiling | Configuration | best val NLL | delta vs independent |",
        "|---|---|---|---|---|---|",
    ]
    for _, r in table.iterrows():
        lines.append(
            f"| {r['basket']} | {r['mean_pairwise_corr']:.2f} | {r['corr_ceiling']:.2f} | {r['label']} | "
            f"{r['nll_mean']:.3f} +/- {r['nll_std']:.3f} | {r['delta_mean']:+.3f} +/- {r['delta_std']:.3f} |"
        )
    lines += [
        "",
        "Figure: `exp3_basket_scaling.png`. Data: `exp3_basket_comparison.csv`.",
        "",
        "Caveats: ~400 train / ~100 val windows per basket; deltas within one seed-std",
        "of zero should be read as no measurable effect.",
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


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(name)s | %(levelname)s | %(message)s")
    main_compare_baskets()
