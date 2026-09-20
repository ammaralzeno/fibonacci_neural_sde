import torch
from torch import nn
from torchmetrics import MeanAbsoluteError

from amgm.models.mlp_multi import NeuralSDEMoEMultiAsset

"""Synthetic multi-asset Neural SDE for correlated GBM recovery test.

Simulated process (Euler-Maruyama on log-prices, implemented in SyntheticCorrelatedGBMDataset):
    dX_i = mu_i * X_i * dt + sigma_i * X_i * (L_R dW)_i,

with equicorrelation R = (1 - rho) * I + rho * 11'. The ground-truth R is known,
so the run reports how well the model's learned correlation matrix recovers it.
"""


def get_trainer_cfg():
    n_assets = 5

    run_cfg = dict(
        run_idx=1,
        run_name="synthetic_gbm_multi_recovery",
        rng_seed=None,  # set to an int for reproducible runs, or None for a fresh random seed each run
        batch_size=128,
        num_workers=0,
        max_epochs=20,
    )

    dset_cfg = dict(
        training_data_type="synthetic_gbm_multi",
        n_assets=n_assets,
        lookback_window=64,
        # dt in: X_{t+1} = X_t + mu(X_t) * dt + sigma(X_t) * sqrt(dt) * (L_R dW)_t
        dt=1.0 / 252.0,
        num_paths=384,
        steps_per_path=48,
        train_split=0.8,
        rho=0.4,
    )

    model_cfg = dict(
        _target_=NeuralSDEMoEMultiAsset,
        lookback_window=dset_cfg["lookback_window"],
        n_assets=n_assets,  # overwritten at runtime with the actual basket size
        num_features=7,
        hidden_sizes=[64, 32],
        corr_init=None,     # overwritten at runtime with the increment correlation of the training data
    )

    trainer_cfg = dict(
        run_cfg=run_cfg,
        dset_cfg=dset_cfg,
        model_cfg=model_cfg,
        loss_cfg=dict(_target_=nn.MSELoss),
        acc_cfg=dict(_target_=MeanAbsoluteError),
        optim_cfg=dict(_target_=torch.optim.Adam, lr=5e-4, weight_decay=1e-6),
        sched_cfg=None,
        entropy_beta=0.1,
        entropy_beta_min=1e-2,
        entropy_beta_warmup_steps=200,
        entropy_beta_decay_steps=800,
        expert_balance_lambda=0,
        compile_model=False,  # CPU training: torch.compile needs Python dev headers and gains little for this model size
    )

    return trainer_cfg
