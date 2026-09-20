import argparse
import importlib
import multiprocessing
import warnings
from pathlib import Path
import numpy as np
import torch
from lightning import Trainer, seed_everything
from lightning.pytorch import loggers as pl_loggers
from lightning.pytorch.callbacks import ModelCheckpoint
import logging
from torch.utils.data import DataLoader, random_split
import re

from amgm import config as amgm_config
from amgm.data.neural_SDE_multi import MultiAssetNeuralSDEDataset, SyntheticCorrelatedGBMDataset
from amgm.models.neural_SDE.multi_runner import MultiAssetNeuralSDERunner
import amgm.utils.common as common
import amgm.utils.myplot as myplot

# For parallel run is necessary otherwise, torch will overload each vCPU 
# torch.set_num_threads(1)          # limit PyTorch to 1 thread for intra-op parallelism
# torch.set_num_interop_threads(1)  # limit inter-op parallelism to 1 thread
torch.set_num_threads(16)
torch.set_num_interop_threads(2)

def _resolve_seed(seed_value):
    if seed_value in (None, "random"):
        return torch.seed() % (2**31 - 1)  
    return int(seed_value)

def _build_dataloaders(dset_cfg, batch_size, seed):

    if dset_cfg.get("training_data_type") == "synthetic_gbm_multi":
        dataset = SyntheticCorrelatedGBMDataset(**dset_cfg, rng_seed=seed)
        is_synthetic_dataset = True
    else:
        dataset = MultiAssetNeuralSDEDataset(**dset_cfg, rng_seed=seed)
        is_synthetic_dataset = False
        
    train_ratio = float(dset_cfg.get("train_split", 0.8))
    train_size = max(1, int(train_ratio * len(dataset)))
    val_size = len(dataset) - train_size

    split_gen = torch.Generator().manual_seed(seed)
    train_set, val_set = random_split(dataset, [train_size, val_size], generator=split_gen)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, dataset, is_synthetic_dataset, seed


def _format_validation_metrics(val_metrics):
    if not val_metrics:
        return "Validation metrics: n/a"

    metrics = val_metrics[0] if isinstance(val_metrics, list) else val_metrics

    pi_mean_values = []
    pi_var_values = []
    other_metrics = []

    def _metric_sort_key(item):
        key, _ = item
        match = re.search(r"_e(\d+)$", key)
        if match:
            return int(match.group(1))
        return key

    def _pretty_number(value):
        value = float(value)
        if abs(value) < 5e-4:
            return "0"
        return f"{value:.3f}"

    for key, value in metrics.items():
        if key.startswith("val/pi_mean_e"):
            pi_mean_values.append((key, value))
        elif key.startswith("val/pi_var_e"):
            pi_var_values.append((key, value))
        else:
            other_metrics.append((key, value))

    pi_mean_values.sort(key=_metric_sort_key)
    pi_var_values.sort(key=_metric_sort_key)
    other_metrics.sort(key=lambda item: item[0])

    lines = ["Validation metrics:"]

    for key, value in other_metrics:
        pretty_key = key.removeprefix("val/")
        if value is None:
            pretty_value = "n/a"
        elif isinstance(value, (float, int)):
            pretty_value = _pretty_number(value)
        else:
            pretty_value = str(value)
        lines.append(f"  {pretty_key}: {pretty_value}")

    if pi_mean_values:
        pi_mean = ", ".join(_pretty_number(value) for _, value in pi_mean_values)
        lines.append(f"  pi = [{pi_mean}]")

    if pi_var_values:
        pi_var = ", ".join(_pretty_number(value) for _, value in pi_var_values)
        lines.append(f"  pi_var = [{pi_var}]")

    return "\n".join(lines)


def _learned_correlation_matrix(mdl):
    """Current learned correlation matrix of the model (unwraps torch.compile if present)."""
    model = getattr(mdl.model, "_orig_mod", mdl.model)
    return model.correlation_matrix().detach().cpu().numpy()


def _evaluate_correlation_recovery(learned_corr, dataset):
    """Compare the learned correlation matrix with the synthetic ground truth."""
    true_corr = getattr(dataset, "true_corr", None)
    if true_corr is None:
        return None
    true = np.asarray(true_corr)
    off_diag = ~np.eye(len(true), dtype=bool)
    return {
        "corr_offdiag_mae": float(np.abs(learned_corr[off_diag] - true[off_diag]).mean()),
        "corr_offdiag_max_err": float(np.abs(learned_corr[off_diag] - true[off_diag]).max()),
        "corr_frobenius_err": float(np.linalg.norm(learned_corr - true)),
    }


def main(trainer_cfg, save_rollout_plots, model_type):

    # Suppress Lightning warning about num_workers=0
    warnings.filterwarnings("ignore", ".*does not have many workers which may be a bottleneck.*")
    multiprocessing.set_start_method("spawn", force=True)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    
    wdir = amgm_config.work_dir("neural_SDE")

    dset_cfg = trainer_cfg["dset_cfg"]
    run_cfg = trainer_cfg["run_cfg"]
    batch_size = run_cfg["batch_size"]
    seed = _resolve_seed(run_cfg.get("rng_seed"))
    seed_everything(seed, workers=True)
        
    version = run_cfg.get("run_name", "US_Stocks_Multi")
    logger = pl_loggers.TensorBoardLogger(
        name=Path(__file__).stem,
        save_dir=wdir / "logs",
        version=version,
    )
    # Persist per-epoch metrics (val/loss, val/loss_sde, ...) to metrics.csv in the
    # same log dir for the Experiment 3 ablation table and convergence curves.
    csv_logger = pl_loggers.CSVLogger(
        save_dir=wdir / "logs",
        name=Path(__file__).stem,
        version=version,
    )
    
    checkpoint_callback = ModelCheckpoint(
            monitor="val/loss",
            mode="min",
            auto_insert_metric_name=False,  # Avoid "val/loss" key be part of the filename, explicitly add the "val_loss" to the filename instead
            filename="best-{epoch:02d}-val_loss={val/loss:.4f}_NEW_" + model_type,  # Explicitly named metric val_loss
            save_top_k=1,                # Only save the best
            dirpath=Path(logger.log_dir) / "checkpoints" # Your directory for checkpoints
            )
    
    print(f"Working directory: {wdir}")
    print(f"Relative log path: {Path(logger.log_dir).relative_to(wdir)}")
    print(f"Full log path: {logger.log_dir}")

    from time import time

    t0 = time()

    # The dataset is built before the runner: the basket size and the increment
    # correlation of the training data are data-dependent model inputs.
    train_loader, val_loader, dataset, is_synthetic_dataset, seed = _build_dataloaders(dset_cfg, batch_size, seed)
    print(f"Using rng_seed={seed}")

    model_cfg = trainer_cfg["model_cfg"]
    model_cfg["n_assets"] = dataset.n_assets
    model_cfg["corr_init"] = dataset.asset_corr_init.tolist()

    mdl = MultiAssetNeuralSDERunner(**trainer_cfg)

    trainer = Trainer(
        max_epochs=run_cfg["max_epochs"],
        check_val_every_n_epoch=1,
        logger=csv_logger,
        accelerator="cpu",
        callbacks=[checkpoint_callback]
    )
    trainer.fit(mdl, train_dataloaders=train_loader, val_dataloaders=val_loader)

    print(f"Training completed in {time() - t0:.2f} seconds.")

    val_metrics = trainer.validate(mdl, dataloaders=val_loader, verbose=False)
    predictions = trainer.predict(mdl, dataloaders=val_loader)

    print(_format_validation_metrics(val_metrics))

    learned_corr = _learned_correlation_matrix(mdl)
    print(f"Learned correlation matrix ({dataset.n_assets} assets):")
    print(np.round(learned_corr, 3))

    if predictions and isinstance(predictions[0], dict):
        first_batch = predictions[0]
        print(f"Prediction batch keys: {list(first_batch.keys())}")
        print(f"Predicted next-price batch shape: {first_batch['x_tp1_pred'].shape}")
        
        if save_rollout_plots:
            # Randomly plot plot_fib_levels for 5 random samples x 2 assets from the validation set
            random_indices = torch.randperm(first_batch["x_window"].shape[0])[:5]
            for idx in random_indices:
                for asset_idx in range(min(2, first_batch["x_window"].shape[-1])):
                    myplot.price_window_and_fib_levels(
                        x_window=first_batch["x_window"][idx, :, asset_idx].cpu().numpy(),
                        fib_levels=first_batch["fib_levels"][idx, asset_idx].cpu().numpy(),
                        features=first_batch["features"][idx, asset_idx].cpu().numpy(),
                        wdir=wdir,
                        x_t=first_batch["x_t"][idx, asset_idx].item(),
                        x_tp1=first_batch["x_tp1"][idx, asset_idx].item(),
                        x_tp1_pred=first_batch["x_tp1_pred"][idx, asset_idx].item(),
                        file_name=f"price_window_and_fib_levels_{idx.item()}_asset{asset_idx}.png"
                    )

    residual_metrics = common.evaluate_residual_calibration(predictions, dset_cfg)
    print("Residual calibration metrics (per-asset, flattened):")
    common.print_dict(residual_metrics)

    multivariate_metrics = common.evaluate_multivariate_residual_calibration(predictions, dset_cfg)
    print("Multivariate residual calibration metrics (whitened by predicted chol_sigma):")
    common.print_dict(multivariate_metrics)

    if is_synthetic_dataset:
        recovery = _evaluate_correlation_recovery(learned_corr, dataset)
        if recovery is not None:
            print("Synthetic correlation recovery metrics:")
            common.print_dict(recovery)

    return val_metrics, predictions

if __name__ == "__main__":
    # This script saves 10 model checkpoints with best val_acc in Path(logger.log_dir) / "checkpoints"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s"
    )

    save_rollout_plots = False
    model_type = "MoE_Multi"
    
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--trainer_cfg",
        help="Dotted module path (relative to this package) exposing get_trainer_cfg().",
    )
    args = parser.parse_args()

    if args.trainer_cfg is None:
        raise ValueError("Please provide a trainer configuration module using --trainer_cfg.")
    
    cfg_path = f"trainer_cfg.neural_SDE.{args.trainer_cfg}"
    cfg_module = importlib.import_module(cfg_path, package=__package__ or "experiments.neural_SDE")
    trainer_cfg = cfg_module.get_trainer_cfg()
   
    main(trainer_cfg, save_rollout_plots, model_type)
