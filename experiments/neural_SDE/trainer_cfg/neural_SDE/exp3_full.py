"""Experiment 3 configuration: full joint model (portfolio context + learned R)."""

from experiments.neural_SDE.trainer_cfg.neural_SDE.US_Stocks_Multi import get_trainer_cfg as _base_cfg


def get_trainer_cfg():
    trainer_cfg = _base_cfg()
    trainer_cfg["run_cfg"] = dict(trainer_cfg["run_cfg"], run_name="exp3_full")
    trainer_cfg["model_cfg"] = dict(trainer_cfg["model_cfg"], use_context=True, learn_corr=True)
    return trainer_cfg
