"""Experiment 3 ablation: portfolio context only (context in gates/experts, R = I)."""

from experiments.neural_SDE.trainer_cfg.neural_SDE.US_Stocks_Multi import get_trainer_cfg as _base_cfg


def get_trainer_cfg():
    trainer_cfg = _base_cfg()
    trainer_cfg["run_cfg"] = dict(trainer_cfg["run_cfg"], run_name="exp3_context_only")
    trainer_cfg["model_cfg"] = dict(trainer_cfg["model_cfg"], use_context=True, learn_corr=False)
    return trainer_cfg
