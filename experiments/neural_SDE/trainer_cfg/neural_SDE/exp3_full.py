"""Experiment 3 configuration: full joint model (portfolio context + learned R).

Part of the Exp 3 ablation family on a fixed 5-asset basket with rng_seed=42:
exp3_independent / exp3_corr_only / exp3_context_only / exp3_full.
The base US_Stocks_Multi config is loaded by file path so this module also works
when loaded standalone via importlib.spec_from_file_location (tests).
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
    trainer_cfg["run_cfg"] = dict(trainer_cfg["run_cfg"], run_name="exp3_full", rng_seed=42)
    trainer_cfg["dset_cfg"] = dict(trainer_cfg["dset_cfg"], n_assets=5)
    trainer_cfg["model_cfg"] = dict(
        trainer_cfg["model_cfg"], n_assets=5, use_context=True, learn_corr=True
    )
    return trainer_cfg
