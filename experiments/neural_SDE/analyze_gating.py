"""Offline entry point: python -m experiments.neural_SDE.analyze_gating --help."""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, help="MoE checkpoint (defaults to training run's best)")
    parser.add_argument("--training-run", type=Path, help="Directory with collection manifest and epoch tables")
    parser.add_argument("--issue-ids-file", type=Path, help="One stock ID per line; required if checkpoint lacks selected IDs")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--data-path", type=Path, help="Override security_data.txt location")
    parser.add_argument("--seed", type=int, default=1, help="Local bootstrap seed")
    parser.add_argument("--render-only", action="store_true", help="Rebuild figures from an existing output directory")
    args = parser.parse_args()
    if args.render_only:
        if args.output_dir is None:
            parser.error("--render-only requires --output-dir")
        from experiments.neural_SDE.gating_diagnostics.plots import render_report
        render_report(args.output_dir)
        return
    checkpoint = args.checkpoint
    if args.training_run:
        manifest = json.loads((args.training_run / "manifest.json").read_text(encoding="utf-8"))
        checkpoint = checkpoint or manifest.get("checkpoint")
        output_dir = args.output_dir or args.training_run
    else:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        output_dir = args.output_dir or Path("workspace/neural_SDE/gating_diagnostics") / f"analysis_{stamp}"
        if output_dir.exists():
            parser.error("Checkpoint-only analysis requires a fresh output directory")
        print("Experiment 1 unavailable: historical gating probabilities cannot be reconstructed from a checkpoint.")
    if not checkpoint:
        parser.error("Provide --checkpoint or a completed --training-run")
    ids = None
    if args.issue_ids_file:
        ids = [line.strip() for line in args.issue_ids_file.read_text().splitlines() if line.strip()]
    from experiments.neural_SDE.gating_diagnostics.analysis import analyze_checkpoint
    report = analyze_checkpoint(checkpoint, output_dir, issue_ids=ids, seed=args.seed,
                                data_path=args.data_path, training_run=args.training_run)
    print(f"Report: {report.resolve()}")


if __name__ == "__main__":
    main()
