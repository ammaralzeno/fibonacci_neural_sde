import argparse
import importlib
import logging
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from lightning import seed_everything

from amgm import config as amgm_config
from amgm.data.neural_SDE_multi import MultiAssetNeuralSDEDataset, _fib_features_multi
from amgm.models.neural_SDE.multi_runner import MultiAssetNeuralSDERunner
import amgm.utils.common as common

# For parallel run is necessary otherwise, torch will overload each vCPU 
# torch.set_num_threads(1)          # limit PyTorch to 1 thread for intra-op parallelism
# torch.set_num_interop_threads(1)  # limit inter-op parallelism to 1 thread
torch.set_num_threads(16)
torch.set_num_interop_threads(2)

def _resolve_seed(seed_value):
    if seed_value in (None, "random"):
        return torch.seed() % (2**31 - 1)  
    return int(seed_value)


def _load_multi_checkpoint(checkpoint_path):
    """Load a MultiAssetNeuralSDERunner checkpoint (stripping torch.compile prefixes)."""
    checkpoint_path = str(Path(checkpoint_path).expanduser())
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    hparams = checkpoint.get("hyper_parameters")
    if not isinstance(hparams, dict):
        raise RuntimeError("Cannot load checkpoint without hyper_parameters.")

    hparams = dict(hparams)
    hparams["compile_model"] = False  # never compile when loading for generation
    model = MultiAssetNeuralSDERunner(**hparams)
    state_dict = checkpoint.get("state_dict", {})
    if not isinstance(state_dict, dict):
        raise RuntimeError("Checkpoint state_dict is missing or invalid.")

    migrated_state = {}
    for key, value in state_dict.items():
        migrated_key = key
        if migrated_key.startswith("model._orig_mod."):
            migrated_key = migrated_key.replace("model._orig_mod.", "model.", 1)
        migrated_state[migrated_key] = value

    model.load_state_dict(migrated_state, strict=True)
    if any(key.startswith("model._orig_mod.") for key in state_dict):
        logging.warning(
            "Loaded compiled checkpoint after stripping model._orig_mod.* prefixes."
        )
    return model


def _compute_min_range_multi(x_window):
    """Per-asset min/range of (batch, w, N) windows; mirrors _compute_min_range in generate_samples.py."""
    x_min = x_window.amin(dim=1, keepdim=True)
    x_max = x_window.amax(dim=1, keepdim=True)
    x_range = torch.clamp(x_max - x_min, min=1e-8)
    return x_min, x_range


def _rollout_sde_multi(model, batch, n_steps, dt, seed):
    """Correlated Monte Carlo rollout of the joint SDE.

    x_{t+1} = x_t + mu * dt + (chol_sigma @ z) * sqrt(dt),  z ~ N(0, I_N)

    where chol_sigma is the model's Cholesky factor of Sigma_t = diag(sigma_t) R diag(sigma_t),
    so each step samples correlated noise across the N assets. The rolling window is
    re-normalized per asset before every inference, matching training-time preprocessing.

    Returns:
        synthetic_paths: (batch, n_steps + 1, N) in original price scale
        pi_paths:        (batch, n_steps, N, 3) gate probabilities along the path
    """
    gen = torch.Generator().manual_seed(int(seed))

    x_window_original = batch["price_window_original"].detach().clone().float()  # (batch, w, N)

    # Keep one rollout per sample: shape [batch_size, n_steps + 1, N]
    synthetic_path = [x_window_original[:, -1, :].detach().cpu().numpy()]  # For continuous plot visualization
    pi_path = []
    with torch.no_grad():
        for step_idx in range(n_steps):
            # Re-normalize each rolling window per asset before inference, matching training-time preprocessing.
            x_min, x_range = _compute_min_range_multi(x_window_original)
            x_window = (x_window_original - x_min) / x_range

            f_t, _ = _fib_features_multi(x_window.numpy())
            f_t = torch.from_numpy(f_t)
            mu, chol_sigma, pi = model(x_window, f_t)

            z = torch.randn(mu.shape, generator=gen, dtype=mu.dtype, device=mu.device)  # (batch, N)
            noise = (chol_sigma @ z.unsqueeze(-1)).squeeze(-1)  # (batch, N) correlated across assets
            x_next_norm = x_window[:, -1, :] + mu * dt + noise * (dt ** 0.5)
            x_next_original = x_next_norm * x_range[:, 0, :] + x_min[:, 0, :]

            synthetic_path.append(x_next_original.detach().cpu().numpy())
            pi_path.append(pi.detach().cpu().numpy())
            x_window_original = torch.cat([x_window_original[:, 1:, :], x_next_original.unsqueeze(1)], dim=1)

    return (
        np.stack(synthetic_path, axis=1).astype(np.float32),
        np.stack(pi_path, axis=1).astype(np.float32),
    )


def _select_forecast_origins(dataset, n_conditions, seed):
    """Seeded random selection of distinct forecast origins (dataset indices)."""
    n = len(dataset)
    if n_conditions > n:
        logging.warning(
            "Only %s windows available; requested %s forecast origins.",
            n, n_conditions,
        )
        n_conditions = n
    generator = torch.Generator().manual_seed(int(seed))
    return torch.randperm(n, generator=generator)[:n_conditions].tolist()


def fetch_original_future_multi(dataset, test_date, n_steps):
    """Real basket prices from the forecast origin onwards, NaN-padded to (n_steps+1, N)."""
    origin_idx = dataset.dates.index(str(test_date))
    future = dataset.prices_aligned[origin_idx : origin_idx + 1 + n_steps]
    out = np.full((n_steps + 1, dataset.n_assets), np.nan, dtype=np.float32)
    out[: len(future)] = future
    return out


def _daily_returns(paths):
    """Simple daily returns of (..., T, N) price paths -> (..., T-1, N)."""
    return (paths[..., 1:, :] - paths[..., :-1, :]) / np.maximum(paths[..., :-1, :], 1e-8)


def _plot_correlation_sanity(real_corr, synth_corr, output_file, basket_issue_ids):
    """Real vs synthetic daily-return correlation heatmaps + absolute difference."""
    diff = np.abs(real_corr - synth_corr)
    off_diag = ~np.eye(len(real_corr), dtype=bool)
    mae = diff[off_diag].mean()

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.6))
    for ax, mat, title, vmin, vmax in [
        (axes[0], real_corr, "Real daily-return correlation", -1, 1),
        (axes[1], synth_corr, "Synthetic (MC rollouts) daily-return correlation", -1, 1),
        (axes[2], diff, f"|difference|  (off-diag MAE={mae:.3f})", 0, 1),
    ]:
        im = ax.imshow(mat, vmin=vmin, vmax=vmax, cmap="RdBu_r")
        ax.set_title(title, fontsize=10)
        ax.set_xticks(range(len(basket_issue_ids)), basket_issue_ids, rotation=90, fontsize=7)
        ax.set_yticks(range(len(basket_issue_ids)), basket_issue_ids, fontsize=7)
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle("Multi-asset Neural SDE: correlation sanity check", fontsize=12)
    fig.tight_layout()
    fig.savefig(output_file, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return mae


def _plot_mc_ensemble_multi(seed_window, synthetic_paths, original_future, test_date, basket_issue_ids, output_plot):
    """One panel per asset: seed window + MC ensemble + median/5--95% interval + real future."""
    n_assets = seed_window.shape[-1]
    fig, axes = plt.subplots(1, n_assets, figsize=(4.2 * n_assets, 4.5), sharex=True)
    if n_assets == 1:
        axes = [axes]
    w = seed_window.shape[0]
    hist_idx = np.arange(w)
    future_idx = np.arange(w - 1, w - 1 + synthetic_paths.shape[1])

    for a, ax in enumerate(axes):
        ax.plot(hist_idx, seed_window[:, a], color="black", linewidth=1.5, label="seed window")
        for path_idx in range(synthetic_paths.shape[0]):
            ax.plot(
                future_idx, synthetic_paths[path_idx, :, a],
                color="tab:blue", linewidth=0.9, alpha=0.18,
                label="MC paths" if path_idx == 0 else None,
            )
        lower, upper = np.quantile(synthetic_paths[:, :, a], [0.05, 0.95], axis=0)
        ax.fill_between(future_idx, lower, upper, color="tab:blue", alpha=0.16, label="MC 5--95% interval")
        ax.plot(future_idx, np.median(synthetic_paths[:, :, a], axis=0), color="navy", linewidth=2, label="MC median")
        original = original_future[:, a]
        valid = np.isfinite(original)
        if valid.any():
            ax.plot(future_idx[valid], original[valid], color="tab:green", linewidth=2, label="original price")
        ax.set_title(f"asset {a}: {basket_issue_ids[a]}", fontsize=10)
        if a == 0:
            ax.legend(fontsize=8, loc="best")
        ax.grid(True, alpha=0.3)
    fig.suptitle(f"Multi-asset Neural SDE MC ensemble, forecast origin: {test_date}", fontsize=12)
    fig.tight_layout()
    fig.savefig(output_plot, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main_mc_multi_asset(
    trainer_cfg,
    checkpoint_path,
    n_conditions,
    mc_paths,
    n_steps,
    output_dir,
    seed_override=None,
    dataset=None,
    artifacts_dir=None,
):
    """Generate correlated MC rollouts for several forecast origins of the basket.

    Artifacts (schema agreed with Member 4):
      - Data/Synthetic/synthetic_rollout_MC_multi_valid_samples.npz
      - Data/Synthetic/synthetic_rollout_paths_MultiMoE.csv
      - Data/Synthetic/correlation_sanity_multi.png

    dataset/artifacts_dir are injectable for testing; defaults build the real
    dataset from trainer_cfg and write to Data/Synthetic.
    """
    if n_conditions <= 0 or mc_paths <= 0:
        raise ValueError("n_conditions and mc_paths must be positive.")

    run_cfg = trainer_cfg["run_cfg"]
    dset_cfg = dict(trainer_cfg["dset_cfg"])
    dset_cfg["max_windows"] = None  # Do not truncate windows; origins are subsampled afterwards.

    configured_seed = seed_override if seed_override is not None else run_cfg.get("rng_seed")
    seed = _resolve_seed(configured_seed)
    seed_everything(seed, workers=True)

    if dataset is None:
        dataset = MultiAssetNeuralSDEDataset(**dset_cfg, rng_seed=seed)
    n_assets = dataset.n_assets
    basket = dataset.basket_issue_ids

    origin_indices = _select_forecast_origins(dataset, n_conditions, seed)
    base_samples = [dataset[idx] for idx in origin_indices]
    print(
        f"MC multi mode: basket={n_assets} assets, initial_conditions={len(base_samples)}, "
        f"mc_paths_per_condition={mc_paths}, total_paths={len(base_samples) * mc_paths}, "
        f"n_steps={n_steps}, master_seed={seed}"
    )

    # Reconstruct original-scale seed windows from the stored normalization
    seed_windows = np.stack(
        [
            (s.price_window * s.sample_range.unsqueeze(0) + s.sample_min.unsqueeze(0)).numpy()
            for s in base_samples
        ]
    ).astype(np.float32)  # (n_conditions, w, N)

    # Vectorized batch: every forecast origin repeated mc_paths times
    batch = {
        "price_window_original": torch.from_numpy(seed_windows).repeat_interleave(mc_paths, dim=0).float(),
    }

    model = _load_multi_checkpoint(checkpoint_path)
    model.eval()

    dt = float(dset_cfg.get("dt", 1.0))
    synthetic_paths, pi_paths = _rollout_sde_multi(
        model,
        batch,
        n_steps=n_steps,
        dt=dt,
        seed=seed,
    )

    # A path is valid only if every asset's path passes the 1D sanity check
    is_valid = np.asarray(
        [
            all(bool(common.sanity_check_path(path[:, a])) for a in range(n_assets))
            for path in synthetic_paths
        ],
        dtype=bool,
    )

    test_dates = [str(s.test_dates) for s in base_samples]
    original_futures = np.stack(
        [fetch_original_future_multi(dataset, td, n_steps) for td in test_dates]
    )  # (n_conditions, n_steps+1, N), NaN-padded near the end of the calendar

    # Fibonacci levels at each forecast origin, denormalized to original price scale
    fib_levels_orig = np.stack(
        [
            (s.fib_levels * s.sample_range.unsqueeze(-1) + s.sample_min.unsqueeze(-1)).numpy()
            for s in base_samples
        ]
    ).astype(np.float32)  # (n_conditions, N, 7)

    raw_model = getattr(model.model, "_orig_mod", model.model)  # unwrap torch.compile if present
    learned_corr = raw_model.correlation_matrix().detach().cpu().numpy().astype(np.float32)

    # ---------------- NPZ artifact (Member 4 schema) ----------------
    if artifacts_dir is None:
        artifacts_dir = Path(__file__).resolve().parents[2] / "Data" / "Synthetic"
    synthetic_artifacts_dir = Path(artifacts_dir)
    synthetic_artifacts_dir.mkdir(parents=True, exist_ok=True)
    artifacts_file = synthetic_artifacts_dir / "synthetic_rollout_MC_multi_valid_samples.npz"
    np.savez_compressed(
        artifacts_file,
        synthetic_paths=synthetic_paths.astype(np.float32),
        basket_issue_ids=np.asarray(basket, dtype=str),
        seed_windows=seed_windows,
        original_prices=original_futures.astype(np.float32),
        fib_levels=fib_levels_orig,
        test_dates=np.asarray(test_dates, dtype=str),
        is_valid=is_valid,
        corr_matrix=learned_corr,
        master_seed=np.asarray([int(seed)], dtype=np.int64),
    )

    # ---------------- CSV artifact (long format, Member 4 schema) ----------------
    frames = []
    for cond_idx, test_date in enumerate(test_dates):
        dates = pd.bdate_range(start=pd.Timestamp(test_date), periods=n_steps + 1)
        for iteration in range(mc_paths):
            path_idx = cond_idx * mc_paths + iteration
            for a in range(n_assets):
                frames.append(
                    pd.DataFrame(
                        {
                            "IssueId": [
                                f"{basket[a]}_synthetic_MC_{cond_idx:04d}_{iteration:03d}"
                            ]
                            * (n_steps + 1),
                            "SourceIssueId": [str(basket[a])] * (n_steps + 1),
                            "AssetIdx": [a] * (n_steps + 1),
                            "TestDate": [test_date] * (n_steps + 1),
                            "PathId": [path_idx] * (n_steps + 1),
                            "MCIteration": [iteration] * (n_steps + 1),
                            "MasterSeed": [int(seed)] * (n_steps + 1),
                            "IsValid": [bool(is_valid[path_idx])] * (n_steps + 1),
                            "Date": dates,
                            "ClAdjLoc": synthetic_paths[path_idx, :, a],
                        }
                    )
                )
    synthetic_output_file = synthetic_artifacts_dir / "synthetic_rollout_paths_MultiMoE.csv"
    pd.concat(frames, ignore_index=True).to_csv(synthetic_output_file, index=False)

    # ---------------- Correlation sanity plot ----------------
    real_corr = np.corrcoef(_daily_returns(dataset.prices_aligned), rowvar=False).astype(np.float32)
    valid_paths = synthetic_paths[is_valid] if is_valid.any() else synthetic_paths
    synth_returns = _daily_returns(valid_paths).reshape(-1, n_assets)
    synth_corr = np.corrcoef(synth_returns, rowvar=False).astype(np.float32)
    corr_plot_file = synthetic_artifacts_dir / "correlation_sanity_multi.png"
    mae = _plot_correlation_sanity(real_corr, synth_corr, corr_plot_file, basket)

    # ---------------- One ensemble figure per forecast origin ----------------
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for cond_idx, test_date in enumerate(test_dates):
        ensemble_output = output_dir / f"synthetic_rollout_multi_{test_date}_MC.png"
        _plot_mc_ensemble_multi(
            seed_window=seed_windows[cond_idx],
            synthetic_paths=synthetic_paths[cond_idx * mc_paths : (cond_idx + 1) * mc_paths],
            original_future=original_futures[cond_idx],
            test_date=test_date,
            basket_issue_ids=basket,
            output_plot=ensemble_output,
        )

    print(f"Saved MC paths CSV to: {synthetic_output_file}")
    print(f"Saved MC artifacts NPZ to: {artifacts_file}")
    print(f"Saved correlation sanity plot to: {corr_plot_file}")
    print(f"Saved {len(test_dates)} MC ensemble plots under: {output_dir}")
    print(f"Sanity-valid MC paths: {int(is_valid.sum())}/{len(is_valid)}")
    print(f"Correlation sanity: off-diag |real - synthetic| MAE = {mae:.4f}")
    return synthetic_paths


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s"
    )

    parser = argparse.ArgumentParser(
        description=(
            "Generate correlated multi-asset Neural SDE samples. Defaults: 10 forecast "
            "origins, 20 MC paths per origin, 100 forecast steps."
        ),
        epilog=(
            "Example: python experiments/neural_SDE/generate_samples_multi.py "
            "--trainer_cfg US_Stocks_Multi"
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--trainer_cfg",
        default="US_Stocks_Multi",
        help="Dotted module path (relative to this package) exposing get_trainer_cfg().",
    )
    parser.add_argument("--n-conditions", type=int, default=10, help="Number of forecast origins.")
    parser.add_argument("--mc-paths", "--mc_paths", dest="mc_paths", type=int, default=20)
    parser.add_argument("--n-steps", type=int, default=100)
    parser.add_argument(
        "--seed",
        type=int,
        help="Override run_cfg.rng_seed. Recommended for reproducible MC runs.",
    )
    parser.add_argument(
        "--checkpoint-name",
        default=None,
        help="Checkpoint file name; defaults to the single .ckpt in the run's checkpoints dir.",
    )
    args = parser.parse_args()

    cfg_path = f"trainer_cfg.neural_SDE.{args.trainer_cfg}"
    cfg_module = importlib.import_module(cfg_path, package=__package__ or "experiments.neural_SDE")
    trainer_cfg = cfg_module.get_trainer_cfg()

    if args.seed is not None:
        trainer_cfg["run_cfg"] = dict(trainer_cfg["run_cfg"])
        trainer_cfg["run_cfg"]["rng_seed"] = args.seed

    wdir = amgm_config.work_dir("neural_SDE")
    run_name = trainer_cfg["run_cfg"].get("run_name", "US_Stocks_Multi")
    checkpoints_dir = wdir / "logs" / "train_neural_SDE_multi" / run_name / "checkpoints"
    if args.checkpoint_name is not None:
        checkpoint_name = args.checkpoint_name + (".ckpt" if not args.checkpoint_name.endswith(".ckpt") else "")
        checkpoint_path = checkpoints_dir / checkpoint_name
    else:
        available = sorted(checkpoints_dir.glob("*.ckpt"))
        if len(available) != 1:
            raise ValueError(
                f"Expected exactly one checkpoint in {checkpoints_dir}, found {len(available)}. "
                "Pass --checkpoint-name to disambiguate."
            )
        checkpoint_path = available[0]

    output_dir = wdir / (
        f"generated_samples_multi_{args.n_conditions}_origins_{args.mc_paths}_MC_{args.n_steps}_steps"
    )

    start_time = time.time()
    main_mc_multi_asset(
        trainer_cfg,
        checkpoint_path=checkpoint_path,
        n_conditions=args.n_conditions,
        mc_paths=args.mc_paths,
        n_steps=args.n_steps,
        output_dir=output_dir,
        seed_override=args.seed,
    )
    print(f"Execution time: {time.time() - start_time:.2f} seconds")
