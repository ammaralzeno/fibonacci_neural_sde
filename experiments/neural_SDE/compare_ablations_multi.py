"""Experiment 3: compare ablation configurations — validation NLL table + convergence.

Reads each run's metrics.csv (CSVLogger) and checkpoint, and writes the Exp 3
deliverables: exp3_nll_table.csv, exp3_convergence.png, exp3_corr_heatmaps.png,
exp3_findings.md (does joint modeling provide value?).
"""

import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from amgm import config as amgm_config
from experiments.neural_SDE.generate_samples_multi import _load_multi_checkpoint

# Configuration name -> (use_context, learn_corr, display label)
# The *_rel runs repeat the ablation on the relationship-driven basket (exp3_*_rel configs).
EXP3_CONFIGS = {
    "exp3_independent": (False, False, "Independent (no context, R=I)"),
    "exp3_corr_only": (False, True, "Correlation only"),
    "exp3_context_only": (True, False, "Context only (R=I)"),
    "exp3_full": (True, True, "Full joint (context + R)"),
    "exp3_independent_rel": (False, False, "Independent (no context, R=I)"),
    "exp3_corr_only_rel": (False, True, "Correlation only"),
    "exp3_context_only_rel": (True, False, "Context only (R=I)"),
    "exp3_full_rel": (True, True, "Full joint (context + R)"),
}


def load_val_curves(log_root, run_names):
    """Per-epoch validation metrics per run: {name: DataFrame(epoch, val/loss, val/loss_sde)}.

    The post-fit trainer.validate() re-logs the final epoch; dedupe keeps the first.
    """
    curves = {}
    for name in run_names:
        metrics_file = Path(log_root) / name / "metrics.csv"
        df = pd.read_csv(metrics_file)
        val = df[["epoch", "val/loss", "val/loss_sde"]].dropna().drop_duplicates(subset="epoch", keep="first")
        curves[name] = val.reset_index(drop=True)
    return curves


def build_nll_table(curves):
    """One row per configuration: best/final validation NLL and delta vs independent."""
    rows = []
    for name, val in curves.items():
        use_context, learn_corr, label = EXP3_CONFIGS.get(name, (None, None, name))
        best_idx = val["val/loss_sde"].idxmin()
        rows.append(
            {
                "config": name,
                "label": label,
                "use_context": use_context,
                "learn_corr": learn_corr,
                "best_val_nll": float(val.loc[best_idx, "val/loss_sde"]),
                "best_epoch": int(val.loc[best_idx, "epoch"]),
                "final_val_nll": float(val["val/loss_sde"].iloc[-1]),
                "best_val_loss": float(val["val/loss"].min()),
            }
        )
    table = pd.DataFrame(rows)
    indep = table[(table["use_context"] == False) & (table["learn_corr"] == False)]
    if len(indep):
        base = indep["best_val_nll"].iloc[0]
        table["delta_vs_independent"] = table["best_val_nll"] - base  # negative = better than independent
    return table


def plot_convergence(curves, output_file):
    """Validation NLL (val/loss_sde) vs epoch, one curve per configuration."""
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for name, val in curves.items():
        label = EXP3_CONFIGS.get(name, (None, None, name))[2]
        ax.plot(val["epoch"], val["val/loss_sde"], marker="o", markersize=3, linewidth=1.4, label=label)
    ax.set_xlabel("epoch")
    ax.set_ylabel("validation NLL (val/loss_sde)")
    ax.set_title("Experiment 3: joint-architecture ablations — validation NLL convergence", fontsize=11)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_file, dpi=150, bbox_inches="tight")
    plt.close(fig)


def load_learned_corr(checkpoint_path):
    """Learned correlation matrix R from a run checkpoint (identity for learn_corr=False)."""
    runner = _load_multi_checkpoint(checkpoint_path)
    return runner.model.correlation_matrix().detach().cpu().numpy()


def plot_corr_heatmaps(corrs, output_file):
    """Learned correlation matrix per configuration (1 x N panels)."""
    names = list(corrs)
    fig, axes = plt.subplots(1, len(names), figsize=(3.4 * len(names), 3.6))
    axes = np.atleast_1d(axes)
    for ax, name in zip(axes, names):
        corr = corrs[name]
        im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
        label = EXP3_CONFIGS.get(name, (None, None, name))[2]
        off = ~np.eye(corr.shape[0], dtype=bool)
        ax.set_title(f"{label}\noff-diag |mean|={np.abs(corr[off]).mean():.3f}", fontsize=9)
        ax.set_xticks(range(corr.shape[0]), range(corr.shape[0]), fontsize=7)
        ax.set_yticks(range(corr.shape[0]), range(corr.shape[0]), fontsize=7)
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle("Learned cross-asset correlation R per configuration", fontsize=11)
    fig.tight_layout()
    fig.savefig(output_file, dpi=150, bbox_inches="tight")
    plt.close(fig)


def write_findings(output_dir, table):
    """Slide-ready Exp 3 findings: NLL table + mechanism attribution."""
    def nll(use_context, learn_corr):
        row = table[(table["use_context"] == use_context) & (table["learn_corr"] == learn_corr)]
        return float(row["best_val_nll"].iloc[0])

    lines = [
        "# Experiment 3 — Joint architecture ablations: findings",
        "",
        "Same 5-asset basket, same seed (42), 20 epochs each. Validation NLL is the",
        "multivariate Gaussian NLL (`val/loss_sde`); lower is better.",
        "",
        "| Configuration | context | learned R | best val NLL | best epoch | final val NLL | Δ vs independent |",
        "|---|---|---|---|---|---|---|",
    ]
    for _, r in table.iterrows():
        delta = f"{r['delta_vs_independent']:+.4f}" if "delta_vs_independent" in table else "—"
        lines.append(
            f"| {r['label']} | {r['use_context']} | {r['learn_corr']} | "
            f"{r['best_val_nll']:.4f} | {r['best_epoch']} | {r['final_val_nll']:.4f} | {delta} |"
        )
    lines += [
        "",
        "## Mechanism attribution (best val NLL differences, nats)",
        "",
        f"- Correlation channel: independent -> corr-only = {nll(False, True) - nll(False, False):+.4f}",
        f"- Context channel: independent -> context-only = {nll(True, False) - nll(False, False):+.4f}",
        f"- Both (full joint): independent -> full = {nll(True, True) - nll(False, False):+.4f}",
        "",
        "Figures: `exp3_convergence.png`, `exp3_corr_heatmaps.png`. Data: `exp3_nll_table.csv`.",
        "",
        "Caveats: single seed; small sample (~380 train / ~95 val windows). The two Exp 3",
        "baskets bracket dependency strength: coverage-selected (mean pairwise return corr",
        "0.08) vs relationship-selected same-sector (0.84) — joint modeling provides value",
        "only when true cross-asset dependencies exist.",
    ]
    (Path(output_dir) / "exp3_findings.md").write_text("\n".join(lines) + "\n")


def main_compare_ablations(log_root=None, output_dir=None, run_names=None):
    """Build all Exp 3 deliverables from the finished ablation runs."""
    if log_root is None:
        log_root = amgm_config.work_dir("neural_SDE") / "logs" / "train_neural_SDE_multi"
    if output_dir is None:
        output_dir = amgm_config.work_dir("neural_SDE") / "logs" / "exp3_ablations"
    if run_names is None:
        run_names = ["exp3_independent", "exp3_corr_only", "exp3_context_only", "exp3_full"]
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    curves = load_val_curves(log_root, run_names)
    table = build_nll_table(curves)
    table.to_csv(output_dir / "exp3_nll_table.csv", index=False)
    plot_convergence(curves, output_dir / "exp3_convergence.png")

    corrs = {}
    for name in run_names:
        ckpts = sorted((Path(log_root) / name / "checkpoints").glob("*.ckpt"))
        if len(ckpts) != 1:
            raise ValueError(f"Expected exactly one checkpoint for {name}, found {len(ckpts)}.")
        corrs[name] = load_learned_corr(ckpts[0])
    plot_corr_heatmaps(corrs, output_dir / "exp3_corr_heatmaps.png")

    write_findings(output_dir, table)
    print(table.to_string(index=False))
    print(f"Exp 3 deliverables written to: {output_dir}")
    return table


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(name)s | %(levelname)s | %(message)s")

    parser = argparse.ArgumentParser(
        description="Experiment 3: ablation comparison — validation NLL table + convergence curves.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--run-names", nargs="*", default=None,
                        help="Run names under the train log dir; defaults to the four exp3_* configurations.")
    parser.add_argument("--output-dir", default=None,
                        help="Where to write deliverables; defaults to logs/exp3_ablations.")
    args = parser.parse_args()
    main_compare_ablations(run_names=args.run_names, output_dir=args.output_dir)
