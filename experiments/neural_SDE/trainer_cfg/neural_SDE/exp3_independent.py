"""Experiment 3 ablation: independent assets (no portfolio context, R = I).

Equivalent to N independent single-asset models sharing one encoder tower:
no information flows between assets. This is the baseline the joint model
must beat to demonstrate that joint modeling provides value.
"""

import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "US_Stocks_Multi", Path(__file__).with_name("US_Stocks_Multi.py")
)
_base = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_base)


def get_trainer_cfg():
    trainer_cfg = _base.get_trainer_cfg()
    trainer_cfg["run_cfg"] = dict(trainer_cfg["run_cfg"], run_name="exp3_independent", rng_seed=42)
    trainer_cfg["dset_cfg"] = dict(trainer_cfg["dset_cfg"], n_assets=5)
    trainer_cfg["model_cfg"] = dict(
        trainer_cfg["model_cfg"], n_assets=5, use_context=False, learn_corr=False
    )
    return trainer_cfg
