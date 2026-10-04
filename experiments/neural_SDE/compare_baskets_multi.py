"""Experiment 3: compare ablation results across baskets of different dependency strength.

Reads the metrics.csv of every exp3 run (one ablation family per basket) and writes
the cross-basket deliverables to the output dir:
  - exp3_basket_comparison.csv: best validation NLL per basket x configuration
  - exp3_basket_scaling.png: NLL gain vs dependency strength, one line per configuration
  - exp3_basket_findings.md: slide-ready summary
"""

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from amgm import config as amgm_config
from experiments.neural_SDE.compare_ablations_multi import build_nll_table, load_val_curves

# Basket label -> its four ablation run names (same seed/protocol, different issue_ids)
BASKET_RUNS = {
    "same-sector": ["exp3_independent", "exp3_corr_only", "exp3_context_only", "exp3_full"],
    "concentrated": [f"exp3_{c}_conc" for c in ("independent", "corr_only", "context_only", "full")],
    "diversified": [f"exp3_{c}_div" for c in ("independent", "corr_only", "context_only", "full")],
}


def dependency_strength(log_root, run_name):
    """Mean off-diagonal of the training increment correlation the model was initialized with."""
    # BaseLoader: hparams.yaml contains python/name tags that safe_load rejects
    hparams = yaml.load((Path(log_root) / run_name / "hparams.yaml").read_text(), Loader=yaml.BaseLoader)
    corr = np.asarray(hparams["model_cfg"]["corr_init"], dtype=float)
    return float(corr[~np.eye(len(corr), dtype=bool)].mean())


def build_basket_table(log_root, basket_runs):
    """One row per basket x configuration: best validation NLL and delta vs the basket's independent run."""
    tables = []
    for basket, run_names in basket_runs.items():
        table = build_nll_table(load_val_curves(log_root, run_names))
        table.insert(0, "mean_pairwise_corr", dependency_strength(log_root, run_names[0]))
        table.insert(0, "basket", basket)
        tables.append(table)
    return pd.concat(tables, ignore_index=True)


def plot_basket_scaling(table, output_file):
    """Validation NLL gain vs basket dependency strength, one line per configuration."""
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    for (use_context, learn_corr), grp in table.groupby(["use_context", "learn_corr"]):
        if not (use_context or learn_corr):
            continue  # the independent baseline is the zero line
        grp = grp.sort_values("mean_pairwise_corr")
        ax.plot(grp["mean_pairwise_corr"], grp["delta_vs_independent"], marker="o", label=grp["label"].iloc[0])
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
        "Same ablation protocol (seed 42, 20 epochs) on three baskets of increasing",
        "cross-asset dependency. Lower validation NLL is better; delta is vs the basket's",
        "own independent baseline.",
        "",
        "| Basket | mean pairwise corr | Configuration | best val NLL | delta vs independent |",
        "|---|---|---|---|---|",
    ]
    for _, r in table.iterrows():
        lines.append(
            f"| {r['basket']} | {r['mean_pairwise_corr']:.2f} | {r['label']} | "
            f"{r['best_val_nll']:.4f} | {r['delta_vs_independent']:+.4f} |"
        )
    lines += [
        "",
        "Figure: `exp3_basket_scaling.png`. Data: `exp3_basket_comparison.csv`.",
        "",
        "Caveats: single seed; small sample (~400 train / ~100 val windows per basket).",
    ]
    (Path(output_dir) / "exp3_basket_findings.md").write_text("\n".join(lines) + "\n")


def main_compare_baskets(log_root=None, output_dir=None, basket_runs=None):
    """Build the cross-basket Exp 3 deliverables from the finished ablation runs."""
    if log_root is None:
        log_root = amgm_config.work_dir("neural_SDE") / "logs" / "train_neural_SDE_multi"
    if output_dir is None:
        output_dir = amgm_config.work_dir("neural_SDE") / "logs" / "exp3_ablations"
    if basket_runs is None:
        basket_runs = BASKET_RUNS
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    table = build_basket_table(log_root, basket_runs)
    table.to_csv(output_dir / "exp3_basket_comparison.csv", index=False)
    plot_basket_scaling(table, output_dir / "exp3_basket_scaling.png")
    write_basket_findings(output_dir, table)
    print(table.to_string(index=False))
    print(f"Cross-basket Exp 3 deliverables written to: {output_dir}")
    return table


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(name)s | %(levelname)s | %(message)s")
    main_compare_baskets()
