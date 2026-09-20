"""Experiment 3 ablation: correlation only (no portfolio context, learned R).

Isolates the contribution of the learned constant correlation matrix R in the
diffusion, without any cross-asset information in the drift/gates.
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
    trainer_cfg["run_cfg"] = dict(trainer_cfg["run_cfg"], run_name="exp3_corr_only", rng_seed=42)
    trainer_cfg["dset_cfg"] = dict(trainer_cfg["dset_cfg"], n_assets=5)
    trainer_cfg["model_cfg"] = dict(
        trainer_cfg["model_cfg"], n_assets=5, use_context=False, learn_corr=True
    )
    return trainer_cfg
