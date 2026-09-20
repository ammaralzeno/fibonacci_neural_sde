"""Experiment 3 ablation: portfolio context only (context in gates/experts, R = I).

Isolates the contribution of the shared portfolio context vector (state-dependent
cross-asset channel) without correlated diffusion noise.
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
    trainer_cfg["run_cfg"] = dict(trainer_cfg["run_cfg"], run_name="exp3_context_only", rng_seed=42)
    trainer_cfg["dset_cfg"] = dict(trainer_cfg["dset_cfg"], n_assets=5)
    trainer_cfg["model_cfg"] = dict(
        trainer_cfg["model_cfg"], n_assets=5, use_context=True, learn_corr=False
    )
    return trainer_cfg
