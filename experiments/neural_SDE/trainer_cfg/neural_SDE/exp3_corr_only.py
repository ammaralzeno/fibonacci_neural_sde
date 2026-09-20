"""Experiment 3 ablation: correlation only (no portfolio context, learned R)."""

from experiments.neural_SDE.trainer_cfg.neural_SDE.US_Stocks_Multi import get_trainer_cfg as _base_cfg


def get_trainer_cfg():
    trainer_cfg = _base_cfg()
    trainer_cfg["run_cfg"] = dict(trainer_cfg["run_cfg"], run_name="exp3_corr_only", rng_seed=42)
    trainer_cfg["dset_cfg"] = dict(trainer_cfg["dset_cfg"], n_assets=5)
    trainer_cfg["model_cfg"] = dict(
        trainer_cfg["model_cfg"], n_assets=5, use_context=False, learn_corr=True
    )
    return trainer_cfg
