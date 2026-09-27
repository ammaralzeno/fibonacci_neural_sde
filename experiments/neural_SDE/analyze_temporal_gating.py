"""Experiment 4: compare temporal gating on paired real and synthetic trajectories."""

import argparse
from datetime import datetime, timezone
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-run", type=Path, help="Experiment-2 directory with manifest, segments, and real gates")
    parser.add_argument("--paths-per-origin", type=int, default=100, help="Synthetic paths per matched origin (default: 100)")
    parser.add_argument("--seed", type=int, default=1, help="Master rollout and local bootstrap seed (default: 1)")
    parser.add_argument("--output-dir", type=Path, help="Fresh output directory, or existing results for --render-only")
    parser.add_argument("--data-path", type=Path, help="Override the source security_data.txt path")
    parser.add_argument("--render-only", action="store_true", help="Rebuild five figures and HTML from saved CSV tables")
    args = parser.parse_args()
    if args.render_only:
        if not args.output_dir:
            parser.error("--render-only requires --output-dir")
        from experiments.neural_SDE.gating_diagnostics.temporal_plots import render_temporal_report
        render_temporal_report(args.output_dir)
        return
    if not args.training_run or args.paths_per_origin < 1:
        parser.error("Provide --training-run and a positive --paths-per-origin")
    # This existing module initializes Torch thread settings. Import it only in
    # this standalone process, before inference; library metrics have no such side effects.
    from experiments.neural_SDE.generate_samples import _rollout_sde
    from experiments.neural_SDE.gating_diagnostics.temporal import run_temporal
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    root = args.output_dir or Path("Results/gating_diagnostics") / f"temporal_seed{args.seed}_{stamp}"
    report = run_temporal(args.training_run, root, _rollout_sde, args.paths_per_origin, args.seed, args.data_path)
    print(f"Report: {report}")


if __name__ == "__main__":
    main()
