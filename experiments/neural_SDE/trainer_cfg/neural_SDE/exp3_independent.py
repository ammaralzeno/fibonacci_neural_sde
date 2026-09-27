"""Experiment 3 ablation: independent assets (no portfolio context, R = I)."""

from experiments.neural_SDE.trainer_cfg.neural_SDE.US_Stocks_Multi import get_trainer_cfg as _base_cfg


def get_trainer_cfg():
    trainer_cfg = _base_cfg()
    trainer_cfg["run_cfg"] = dict(trainer_cfg["run_cfg"], run_name="exp3_independent")
    trainer_cfg["model_cfg"] = dict(trainer_cfg["model_cfg"], use_context=False, learn_corr=False)
    return trainer_cfg
