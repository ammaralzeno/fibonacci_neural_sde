"""Experiment 3 ablation on the relationship-driven basket: correlation only."""

from experiments.neural_SDE.trainer_cfg.neural_SDE.exp3_corr_only import get_trainer_cfg as _base_cfg

# Same-sector basket (SectorCode 40), mean pairwise return corr 0.84 in 2014-2017
BASKET = ["00674201", "14335601", "13376801", "01272601", "00764701"]


def get_trainer_cfg():
    trainer_cfg = _base_cfg()
    trainer_cfg["run_cfg"] = dict(trainer_cfg["run_cfg"], run_name="exp3_corr_only_rel")
    trainer_cfg["dset_cfg"] = dict(trainer_cfg["dset_cfg"], issue_ids=list(BASKET))
    return trainer_cfg
