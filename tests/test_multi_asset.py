"""Tests for the multi-asset data pipeline (Phase 1), model (Phase 2),
runner (Phase 3), and training entry/config (Phase 4)."""

import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from amgm.data.neural_SDE_multi import (
    FIB_NUM_LEVELS,
    MultiAssetNeuralSDEDataset,
    MultiAssetNeuralSDESample,
    SyntheticCorrelatedGBMDataset,
    _fib_features_multi,
    _min_max_normalize_multi,
)
from amgm.models.mlp_multi import NeuralSDEMoEMultiAsset
import amgm.utils.common as common
from experiments.neural_SDE import analyze_fib_features_multi as fib_analysis

LOOKBACK = 126
BASKET = ["AAA", "BBB", "CCC"]


def _make_fake_secs(n_days=600, drop_date_for_b=True):
    """Three correlated synthetic assets as an uploaded_df (IssueId/Date/ClAdjLoc)."""
    rng = np.random.default_rng(7)
    dates = pd.bdate_range("2016-01-01", periods=n_days)
    corr = np.array([[1.0, 0.5, 0.0], [0.5, 1.0, -0.2], [0.0, -0.2, 1.0]])
    chol = np.linalg.cholesky(corr)
    rets = rng.standard_normal((n_days, 3)) @ chol.T * 0.01
    prices = 100 * np.exp(np.cumsum(rets, axis=0))

    frames = []
    for j, iid in enumerate(BASKET):
        d, p = dates, prices[:, j]
        if iid == "BBB" and drop_date_for_b:
            keep = d != dates[300]
            d, p = d[keep], p[keep]
        frames.append(pd.DataFrame({"IssueId": iid, "Date": d, "ClAdjLoc": p}))
    return pd.concat(frames, ignore_index=True), dates


def _make_dataset(df, **overrides):
    kwargs = dict(
        issue_ids=list(BASKET),
        start_date="2016-01-01",
        end_date="2018-06-30",
        lookback_window=LOOKBACK,
        uploaded_df=df,
    )
    kwargs.update(overrides)
    return MultiAssetNeuralSDEDataset(**kwargs)


def test_shapes_and_no_nans():
    df, _ = _make_fake_secs()
    ds = _make_dataset(df)
    B = len(ds)
    assert ds.price_window.shape == (B, LOOKBACK, 3)
    assert ds.nxt_prices.shape == (B, 3)
    assert ds.sample_min.shape == (B, 3)
    assert ds.sample_range.shape == (B, 3)
    assert ds.features.shape == (B, 3, FIB_NUM_LEVELS)
    assert ds.fib_levels.shape == (B, 3, FIB_NUM_LEVELS)
    assert ds.asset_corr_init.shape == (3, 3)
    for tensor in (ds.price_window, ds.nxt_prices, ds.features, ds.fib_levels):
        assert torch.isfinite(tensor).all()


def test_inner_join_drops_missing_date():
    df, dates = _make_fake_secs()
    ds = _make_dataset(df)
    missing = dates[300].strftime("%Y-%m-%d")
    expected = [d.strftime("%Y-%m-%d") for d in dates if d.strftime("%Y-%m-%d") != missing]
    assert ds.dates == expected
    assert missing not in ds.test_dates


def test_per_asset_minmax_normalization():
    df, _ = _make_fake_secs()
    ds = _make_dataset(df)
    pw = ds.price_window.numpy()
    assert np.allclose(pw.min(axis=1), 0.0, atol=1e-5)
    assert np.allclose(pw.max(axis=1), 1.0, atol=1e-5)
    assert (ds.sample_range.numpy() > 0).all()


def test_fib_features_match_calculate_fibLevels():
    df, _ = _make_fake_secs()
    ds = _make_dataset(df)
    pw = ds.price_window.numpy()
    B, w, N = pw.shape
    flat = np.transpose(pw, (0, 2, 1)).reshape(B * N, w)
    fib, delta = common.calculate_fibLevels(flat)
    expected_features = ((flat[:, -1:] - fib) / np.maximum(delta, 1e-8)).reshape(B, N, FIB_NUM_LEVELS)
    assert np.allclose(ds.features.numpy(), expected_features, atol=1e-5)
    assert np.allclose(ds.fib_levels.numpy(), fib.reshape(B, N, FIB_NUM_LEVELS), atol=1e-5)


def test_corr_init_is_valid_correlation_matrix():
    df, _ = _make_fake_secs()
    ds = _make_dataset(df)
    R = ds.asset_corr_init
    assert np.allclose(R, R.T, atol=1e-6)
    assert np.allclose(np.diag(R), 1.0, atol=1e-6)
    assert np.linalg.eigvalsh(R).min() > -1e-6
    assert np.abs(R).max() <= 1.0 + 1e-6


def test_max_windows_subsampling_is_deterministic():
    df, _ = _make_fake_secs()
    ds1 = _make_dataset(df, max_windows=50)
    ds2 = _make_dataset(df, max_windows=50)
    assert len(ds1) <= 50
    assert ds1.test_dates == ds2.test_dates


def test_synthetic_gbm_sample_structure():
    syn = SyntheticCorrelatedGBMDataset(n_assets=4, lookback_window=LOOKBACK, num_paths=8, steps_per_path=4)
    B = len(syn)
    assert B == 8 * 4
    assert syn.price_window.shape == (B, LOOKBACK, 4)
    assert syn.features.shape == (B, 4, FIB_NUM_LEVELS)
    assert syn.true_corr.shape == (4, 4)
    sample = syn[0]
    assert len(sample.issue_ids) == 4
    assert torch.isfinite(sample.price_window).all()


def test_synthetic_gbm_recovers_correlation():
    rho = 0.5
    syn = SyntheticCorrelatedGBMDataset(
        n_assets=4, lookback_window=LOOKBACK, num_paths=256, steps_per_path=32, rho=rho, rng_seed=3
    )
    est = syn.asset_corr_init
    true = syn.true_corr
    off_diag = ~np.eye(4, dtype=bool)
    assert np.mean(np.abs(est[off_diag] - true[off_diag])) < 0.1
    assert est[0, 1] > 0.2


# ---------------------------------------------------------------------------
# Phase 2: NeuralSDEMoEMultiAsset model
# ---------------------------------------------------------------------------


class _FixedGate(nn.Module):
    """Stub gate that always returns the same logits, to isolate one expert."""

    def __init__(self, logits):
        super().__init__()
        self.logits = logits

    def forward(self, h):
        return torch.as_tensor(self.logits, dtype=h.dtype).expand(h.shape[0], -1)


def _make_model(n_assets=3, corr_init=None):
    return NeuralSDEMoEMultiAsset(
        lookback_window=LOOKBACK, n_assets=n_assets, hidden_sizes=[32, 16], corr_init=corr_init
    )


def test_model_forward_shapes():
    model = _make_model(n_assets=3)
    x = torch.randn(4, LOOKBACK, 3)
    f = torch.randn(4, 3, FIB_NUM_LEVELS) * 0.1
    mu, chol_sigma, pi = model(x, f)
    assert mu.shape == (4, 3)
    assert chol_sigma.shape == (4, 3, 3)
    assert pi.shape == (4, 3, 3)
    assert torch.allclose(pi.sum(dim=-1), torch.ones(4, 3), atol=1e-6)
    assert torch.isfinite(mu).all() and torch.isfinite(chol_sigma).all()


def test_correlation_matrix_valid_identity_init():
    model = _make_model(n_assets=3)
    R = model.correlation_matrix()
    assert torch.allclose(R, R.T, atol=1e-6)
    assert torch.allclose(R.diagonal(), torch.ones(3), atol=1e-6)
    assert torch.linalg.eigvalsh(R).min() > -1e-6
    assert R.abs().max() <= 1.0 + 1e-6


def test_correlation_matrix_recovers_corr_init():
    corr = torch.tensor([[1.0, 0.5, 0.1], [0.5, 1.0, -0.2], [0.1, -0.2, 1.0]])
    model = _make_model(n_assets=3, corr_init=corr)
    assert torch.allclose(model.correlation_matrix(), corr, atol=1e-4)


def test_chol_sigma_lower_triangular_positive_diag():
    model = _make_model(n_assets=3)
    x = torch.randn(4, LOOKBACK, 3)
    f = torch.randn(4, 3, FIB_NUM_LEVELS) * 0.1
    _, chol_sigma, _ = model(x, f)
    upper = torch.triu(chol_sigma, diagonal=1)
    assert torch.allclose(upper, torch.zeros_like(upper))
    assert (chol_sigma.diagonal(dim1=-2, dim2=-1) > 0).all()


def test_covariance_equals_diag_sigma_R_diag_sigma():
    model = _make_model(n_assets=3)
    x = torch.randn(4, LOOKBACK, 3)
    f = torch.randn(4, 3, FIB_NUM_LEVELS) * 0.1
    _, chol_sigma, _ = model(x, f)
    sigma = chol_sigma.diagonal(dim1=-2, dim2=-1)
    R = model.correlation_matrix()
    Sigma = chol_sigma @ chol_sigma.transpose(-1, -2)
    expected = sigma.unsqueeze(-1) * R * sigma.unsqueeze(-2)
    assert torch.allclose(Sigma, expected, atol=1e-6)


def test_per_asset_equivariance_of_mu_and_pi():
    model = _make_model(n_assets=3).eval()
    x = torch.randn(4, LOOKBACK, 3)
    f = torch.randn(4, 3, FIB_NUM_LEVELS) * 0.1
    perm = [2, 0, 1]
    mu1, _, pi1 = model(x, f)
    mu2, _, pi2 = model(x[:, :, perm], f[:, perm, :])
    assert torch.allclose(mu2, mu1[:, perm], atol=1e-6)
    assert torch.allclose(pi2, pi1[:, perm], atol=1e-6)


def test_bounce_constraint_direction_per_asset():
    model = _make_model(n_assets=2)
    model.gate_head = _FixedGate([20.0, 0.0, 0.0])  # force Bounce expert
    x = torch.randn(8, LOOKBACK, 2)
    f = torch.zeros(8, 2, FIB_NUM_LEVELS)
    f[:, 0, :] = 0.3   # asset 0 above all levels -> bounce pushes up
    f[:, 1, :] = -0.3  # asset 1 below all levels -> bounce pushes down
    mu, _, _ = model(x, f)
    assert (mu[:, 0] > 0).all()
    assert (mu[:, 1] < 0).all()


def test_break_constraint_direction_per_asset():
    model = _make_model(n_assets=2)
    model.gate_head = _FixedGate([0.0, 20.0, 0.0])  # force Break expert
    x = torch.zeros(8, LOOKBACK, 2)
    x[:, -1, 0] = 1.0   # asset 0 last step up -> break continues up
    x[:, -1, 1] = -1.0  # asset 1 last step down -> break continues down
    f = torch.zeros(8, 2, FIB_NUM_LEVELS)
    mu, _, _ = model(x, f)
    assert (mu[:, 0] > 0).all()
    assert (mu[:, 1] < 0).all()


def test_gradient_flows_to_all_parameters():
    model = _make_model(n_assets=3)
    x = torch.randn(4, LOOKBACK, 3)
    f = torch.randn(4, 3, FIB_NUM_LEVELS) * 0.1
    mu, chol_sigma, _ = model(x, f)
    loss = mu.sum() + chol_sigma.sum()
    loss.backward()
    for name, param in model.named_parameters():
        assert param.grad is not None, f"no gradient for {name}"
        assert torch.isfinite(param.grad).all(), f"non-finite gradient for {name}"


# ---------------------------------------------------------------------------
# Phase 3: MultiAssetNeuralSDERunner (multivariate Gaussian NLL)
# ---------------------------------------------------------------------------

from torch.utils.data import DataLoader

from amgm.models.neural_SDE.multi_runner import MultiAssetNeuralSDERunner

GBM_DT = 1.0 / 252.0


def _make_runner(n_assets=4, corr_init=None, lookback=LOOKBACK, **model_overrides):
    model_cfg = dict(
        _target_=NeuralSDEMoEMultiAsset,
        lookback_window=lookback,
        n_assets=n_assets,
        num_features=FIB_NUM_LEVELS,
        hidden_sizes=[32, 16],
        corr_init=corr_init,
    )
    model_cfg.update(model_overrides)
    return MultiAssetNeuralSDERunner(
        run_cfg=dict(batch_size=64),
        dset_cfg=dict(dt=GBM_DT, eps=1e-6),
        model_cfg=model_cfg,
        loss_cfg=dict(_target_=nn.MSELoss),
        acc_cfg=dict(),
        optim_cfg=dict(_target_=torch.optim.Adam, lr=1e-3),
        sched_cfg=None,
        compile_model=False,
    )


def _equicorr(n_assets, rho):
    return (1.0 - rho) * np.eye(n_assets) + rho * np.ones((n_assets, n_assets))


def test_runner_compute_loss_shapes_and_finite():
    syn = SyntheticCorrelatedGBMDataset(n_assets=4, lookback_window=LOOKBACK, num_paths=16, steps_per_path=8)
    runner = _make_runner(n_assets=4, corr_init=syn.asset_corr_init)
    batch = next(iter(DataLoader(syn, batch_size=64)))
    out = runner._compute_loss(batch)
    assert torch.isfinite(out["loss"])
    assert out["mu"].shape == (64, 4)
    assert out["sigma"].shape == (64, 4)
    assert out["chol_sigma"].shape == (64, 4, 4)
    assert out["x_t"].shape == (64, 4)
    assert out["x_tp1_pred"].shape == (64, 4)
    assert out["pi_mean"].shape == (3,)
    assert out["pi_var"].shape == (3,)


def test_multivariate_nll_prefers_true_correlation():
    torch.manual_seed(0)  # deterministic model init: the NLL margin depends on it
    syn = SyntheticCorrelatedGBMDataset(
        n_assets=4, lookback_window=LOOKBACK, num_paths=256, steps_per_path=32, rho=0.4, rng_seed=11
    )
    runner = _make_runner(n_assets=4)
    batch = next(iter(DataLoader(syn, batch_size=4096)))
    x_window, f_t, x_t, x_tp1, _ = runner._prepare_batch(batch)

    def nll_with_corr(corr):
        corr_t = torch.as_tensor(corr, dtype=torch.float32)
        runner.model.chol_corr_param.data = torch.linalg.cholesky(corr_t + 1e-6 * torch.eye(4))
        with torch.no_grad():
            mu, chol_sigma, _ = runner.model(x_window, f_t)
        # mu/sigma do not depend on the correlation parameter, so this isolates R
        return runner._step_nll(x_t, x_tp1, mu, chol_sigma).item()

    nll_true = nll_with_corr(syn.true_corr)
    nll_identity = nll_with_corr(np.eye(4))
    nll_wrong_sign = nll_with_corr(_equicorr(4, -0.2))
    assert nll_true < nll_identity
    assert nll_true < nll_wrong_sign


def test_whitened_residuals_decorrelated_with_true_corr():
    syn = SyntheticCorrelatedGBMDataset(
        n_assets=4, lookback_window=LOOKBACK, num_paths=256, steps_per_path=32, rho=0.4, rng_seed=5
    )
    runner = _make_runner(n_assets=4)
    batch = next(iter(DataLoader(syn, batch_size=8192)))
    x_window, f_t, x_t, x_tp1, _ = runner._prepare_batch(batch)

    def whitened_cross_corr(corr):
        corr_t = torch.as_tensor(corr, dtype=torch.float32)
        runner.model.chol_corr_param.data = torch.linalg.cholesky(corr_t + 1e-6 * torch.eye(4))
        with torch.no_grad():
            mu, chol_sigma, _ = runner.model(x_window, f_t)
        L = chol_sigma * math.sqrt(GBM_DT)
        dx = (x_tp1 - x_t - mu * GBM_DT).unsqueeze(-1)
        z = torch.linalg.solve_triangular(L, dx, upper=False).squeeze(-1).numpy()
        z_corr = np.corrcoef(z, rowvar=False)
        off_diag = z_corr[~np.eye(4, dtype=bool)]
        return np.abs(off_diag).mean()

    # Diagonal scaling does not change correlations, so untrained sigma heads are fine here:
    # whitening with the true R must decorrelate, whitening with identity must not.
    assert whitened_cross_corr(syn.true_corr) < 0.05
    assert whitened_cross_corr(np.eye(4)) > 0.15


def test_entropy_balance_uses_flattened_pi():
    syn = SyntheticCorrelatedGBMDataset(n_assets=3, lookback_window=LOOKBACK, num_paths=8, steps_per_path=4)
    runner = _make_runner(n_assets=3)
    batch = next(iter(DataLoader(syn, batch_size=16)))
    out = runner._compute_loss(batch)
    with torch.no_grad():
        _, _, pi = runner.model(batch.price_window, batch.features)
    pi_flat = pi.reshape(-1, 3)
    assert torch.allclose(out["pi_mean"], pi_flat.mean(dim=0), atol=1e-6)
    expected_entropy = -(pi_flat * torch.log(pi_flat + 1e-8)).sum(dim=-1).mean()
    assert torch.allclose(out["mean_entropy"], expected_entropy, atol=1e-6)


# ---------------------------------------------------------------------------
# Phase 4: training loop smoke test + multivariate calibration
# ---------------------------------------------------------------------------


def test_smoke_training_one_epoch_synthetic_gbm():
    from lightning import Trainer

    syn = SyntheticCorrelatedGBMDataset(n_assets=3, lookback_window=LOOKBACK, num_paths=16, steps_per_path=8)
    runner = _make_runner(n_assets=3, corr_init=syn.asset_corr_init)
    corr_init_chol = runner.model.chol_corr_param.data.clone()
    loader = DataLoader(syn, batch_size=32)
    trainer = Trainer(
        max_epochs=1,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        accelerator="cpu",
    )
    trainer.fit(runner, train_dataloaders=loader, val_dataloaders=loader)
    assert torch.isfinite(runner.trainer.callback_metrics["val/loss"])
    # Lightning clears grads after fit; a moved parameter proves gradient flow
    # through the full training loop (Adam moves params even on tiny grads).
    assert not torch.allclose(runner.model.chol_corr_param.data, corr_init_chol)

    predictions = trainer.predict(runner, dataloaders=loader)
    assert "chol_sigma" in predictions[0]
    assert predictions[0]["chol_sigma"].shape[-2:] == (3, 3)


def test_multivariate_calibration_whitens_with_true_corr():
    syn = SyntheticCorrelatedGBMDataset(
        n_assets=4, lookback_window=LOOKBACK, num_paths=256, steps_per_path=16, rho=0.4, rng_seed=5
    )
    runner = _make_runner(n_assets=4)
    corr_t = torch.as_tensor(syn.true_corr, dtype=torch.float32)
    runner.model.chol_corr_param.data = torch.linalg.cholesky(corr_t + 1e-6 * torch.eye(4))

    loader = DataLoader(syn, batch_size=4096)
    predictions = [runner.predict_step(batch, idx) for idx, batch in enumerate(loader)]

    metrics = common.evaluate_multivariate_residual_calibration(predictions, dict(dt=GBM_DT))
    # Whitening with the true R decorrelates regardless of the (untrained) sigma
    # heads: cross-correlation is invariant to diagonal rescaling.
    assert metrics["z_cross_corr_abs_mean"] < 0.05
    assert metrics["z_cross_corr_abs_max"] < 0.15
    assert metrics["n_residuals"] == len(syn) * 4

    # Identity correlation must fail to decorrelate the same residuals.
    runner.model.chol_corr_param.data = torch.eye(4)
    predictions_id = [runner.predict_step(batch, idx) for idx, batch in enumerate(loader)]
    metrics_id = common.evaluate_multivariate_residual_calibration(predictions_id, dict(dt=GBM_DT))
    assert metrics_id["z_cross_corr_abs_mean"] > 0.15


def test_trainer_cfg_modules_build():
    import importlib.util

    cfg_dir = Path(__file__).resolve().parents[1] / "experiments" / "neural_SDE" / "trainer_cfg" / "neural_SDE"
    for name in ("US_Stocks_Multi", "synthetic_gbm_multi"):
        spec = importlib.util.spec_from_file_location(name, cfg_dir / f"{name}.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        cfg = module.get_trainer_cfg()
        assert cfg["model_cfg"]["_target_"] is NeuralSDEMoEMultiAsset
        assert cfg["model_cfg"]["num_features"] == FIB_NUM_LEVELS
        assert cfg["model_cfg"]["lookback_window"] == cfg["dset_cfg"]["lookback_window"]
        assert "corr_init" in cfg["model_cfg"] and "n_assets" in cfg["model_cfg"]


# ---------------------------------------------------------------------------
# Phase 5: correlated MC generation + Member 4 artifacts
# ---------------------------------------------------------------------------

import importlib.util

_gen_spec = importlib.util.spec_from_file_location(
    "generate_samples_multi",
    Path(__file__).resolve().parents[1] / "experiments" / "neural_SDE" / "generate_samples_multi.py",
)
gen_multi = importlib.util.module_from_spec(_gen_spec)
_gen_spec.loader.exec_module(gen_multi)


class _FakeMultiDataset:
    """Minimal basket dataset with the calendar interface the generator needs.

    Builds windows exactly like MultiAssetNeuralSDEDataset but from an in-memory
    correlated GBM, so tests don't touch the real AM data or BaseAMData.
    """

    def __init__(self, n_assets=3, n_days=220, lookback=40, rho=0.5, seed=0):
        rng = np.random.default_rng(seed)
        corr = _equicorr(n_assets, rho)
        chol = np.linalg.cholesky(corr)
        rets = rng.standard_normal((n_days, n_assets)) @ chol.T * 0.01
        prices = (100 * np.exp(np.cumsum(rets, axis=0))).astype(np.float32)
        self.prices_aligned = prices
        self.dates = pd.bdate_range("2020-01-01", periods=n_days).strftime("%Y-%m-%d").tolist()
        self.n_assets = n_assets
        self.basket_issue_ids = [f"asset{i}" for i in range(n_assets)]
        self.lookback = lookback
        self.lookback_window = lookback  # alias matching MultiAssetNeuralSDEDataset
        self.true_corr = corr.astype(np.float32)

        starts = np.arange(0, n_days - lookback)
        pw = np.stack([prices[s : s + lookback] for s in starts])
        nxt = np.stack([prices[s + lookback] for s in starts])
        norm_windows, norm_nxt, sample_min, sample_range = _min_max_normalize_multi(pw, nxt)
        features, fib_levels = _fib_features_multi(norm_windows)
        self.price_window = torch.from_numpy(norm_windows)
        self.sample_min = torch.from_numpy(sample_min)
        self.sample_range = torch.from_numpy(sample_range)
        self.features = torch.from_numpy(features)
        self.fib_levels = torch.from_numpy(fib_levels)
        self.test_dates = [self.dates[s + lookback - 1] for s in starts]

    def __len__(self):
        return len(self.test_dates)

    def __getitem__(self, idx):
        return MultiAssetNeuralSDESample(
            price_window=self.price_window[idx],
            nxt_price=torch.zeros(self.n_assets),  # not used by the generator
            sample_min=self.sample_min[idx],
            sample_range=self.sample_range[idx],
            features=self.features[idx],
            fib_levels=self.fib_levels[idx],
            test_dates=self.test_dates[idx],
            issue_ids=self.basket_issue_ids,
        )


def test_rollout_recovers_known_correlation():
    n_assets, rho = 4, 0.5
    ds = _FakeMultiDataset(n_assets=n_assets, rho=rho)
    model = NeuralSDEMoEMultiAsset(
        lookback_window=ds.lookback,
        n_assets=n_assets,
        hidden_sizes=[32, 16],
        corr_init=torch.as_tensor(ds.true_corr),
    )
    model.eval()

    # 8 origins x 50 MC paths = 400 rollouts, all in original price scale
    seed_windows = torch.from_numpy(
        np.stack(
            [
                (ds[i].price_window * ds[i].sample_range.unsqueeze(0) + ds[i].sample_min.unsqueeze(0)).numpy()
                for i in range(8)
            ]
        )
    ).float()
    batch = {"price_window_original": seed_windows.repeat_interleave(50, dim=0)}

    paths, pi = gen_multi._rollout_sde_multi(model, batch, n_steps=40, dt=1.0 / 252.0, seed=1)
    assert paths.shape == (400, 41, n_assets)
    assert pi.shape == (400, 40, n_assets, 3)
    assert np.isfinite(paths).all()
    assert (paths > 0).all()

    # The correlated noise (chol_sigma @ z) must imprint R on the rollout returns,
    # even with untrained drift/diffusion heads (drift is O(dt), noise is O(sqrt(dt))).
    rets = gen_multi._daily_returns(paths).reshape(-1, n_assets)
    est = np.corrcoef(rets, rowvar=False)
    off_diag = ~np.eye(n_assets, dtype=bool)
    assert np.abs(est[off_diag] - rho).mean() < 0.1


def test_generation_writes_member4_artifacts(tmp_path):
    n_assets, lookback = 3, 30
    ds = _FakeMultiDataset(n_assets=n_assets, n_days=120, lookback=lookback, rho=0.4, seed=2)
    runner = _make_runner(n_assets=n_assets, corr_init=ds.true_corr, lookback=lookback)

    # Checkpoint round-trip through the same loader the CLI uses
    ckpt_path = tmp_path / "model.ckpt"
    torch.save({"hyper_parameters": dict(runner.hparams), "state_dict": runner.state_dict()}, ckpt_path)

    n_cond, mc, steps = 2, 4, 15
    out_dir = tmp_path / "plots"
    gen_multi.main_mc_multi_asset(
        dict(run_cfg=dict(rng_seed=5), dset_cfg=dict(dt=1.0)),
        checkpoint_path=ckpt_path,
        n_conditions=n_cond,
        mc_paths=mc,
        n_steps=steps,
        output_dir=out_dir,
        dataset=ds,
        artifacts_dir=tmp_path,
    )

    npz = np.load(tmp_path / "synthetic_rollout_MC_multi_valid_samples.npz")
    assert npz["synthetic_paths"].shape == (n_cond * mc, steps + 1, n_assets)
    assert npz["seed_windows"].shape == (n_cond, lookback, n_assets)
    assert npz["original_prices"].shape == (n_cond, steps + 1, n_assets)
    assert npz["fib_levels"].shape == (n_cond, n_assets, FIB_NUM_LEVELS)
    assert npz["corr_matrix"].shape == (n_assets, n_assets)
    assert npz["is_valid"].dtype == bool and len(npz["is_valid"]) == n_cond * mc
    assert list(npz["basket_issue_ids"]) == ds.basket_issue_ids
    assert len(npz["test_dates"]) == n_cond
    assert np.isfinite(npz["synthetic_paths"]).all()

    # Seed windows are in original scale and match the aligned price matrix
    first_date_idx = ds.dates.index(str(npz["test_dates"][0]))
    np.testing.assert_allclose(
        npz["seed_windows"][0],
        ds.prices_aligned[first_date_idx - lookback + 1 : first_date_idx + 1],
        rtol=1e-5,
    )

    csv = pd.read_csv(tmp_path / "synthetic_rollout_paths_MultiMoE.csv")
    expected_cols = {
        "IssueId", "SourceIssueId", "AssetIdx", "TestDate", "PathId",
        "MCIteration", "MasterSeed", "IsValid", "Date", "ClAdjLoc",
    }
    assert expected_cols <= set(csv.columns)
    assert len(csv) == n_cond * mc * n_assets * (steps + 1)
    assert csv["AssetIdx"].nunique() == n_assets
    assert csv["PathId"].nunique() == n_cond * mc

    assert (tmp_path / "correlation_sanity_multi.png").exists()
    assert len(list(out_dir.glob("synthetic_rollout_multi_*_MC.png"))) == n_cond


# ---------------------------------------------------------------------------
# Experiment 2: multi-asset Fibonacci feature analysis
# ---------------------------------------------------------------------------


def test_nearest_level_distances():
    features = np.array([[[0.5, 0.1, -0.05, 0.3, 0.9, 1.2, 1.5]]])  # (1, 1, 7)
    signed, absolute = fib_analysis.nearest_level_distances(features)
    assert signed.shape == (1, 1) and absolute.shape == (1, 1)
    assert absolute[0, 0] == 0.05
    assert signed[0, 0] == -0.05  # signed value of the closest level, not the largest


def test_pairwise_cooccurrence_independence_gives_unit_lift():
    # P(A)=P(B)=1/2, joint exactly 1/4 by construction -> lift exactly 1
    mask = np.array(
        [
            [True, True],
            [True, False],
            [False, True],
            [False, False],
        ]
    )
    out = fib_analysis.pairwise_cooccurrence(mask)
    np.testing.assert_allclose(out["p_near"], [0.5, 0.5])
    assert out["joint"][0, 1] == 0.25
    np.testing.assert_allclose(out["lift"][0, 1], 1.0)
    np.testing.assert_allclose(out["lift"][1, 0], 1.0)


def test_pairwise_cooccurrence_locked_gives_inverse_rate():
    # A and B near on exactly the same days with rate 1/4 -> lift = 1/p = 4
    mask = np.zeros((8, 2), dtype=bool)
    mask[:2, :] = True
    out = fib_analysis.pairwise_cooccurrence(mask)
    np.testing.assert_allclose(out["lift"][0, 1], 4.0)


def test_pairwise_cooccurrence_zero_rate_is_nan():
    mask = np.zeros((8, 3), dtype=bool)
    mask[:2, 0] = True  # asset 1 and 2 never near a level
    out = fib_analysis.pairwise_cooccurrence(mask)
    assert np.isnan(out["lift"][0, 1]) and np.isnan(out["lift"][2, 0])
    assert np.isfinite(out["lift"][0, 0])


def test_next_day_returns_alignment():
    ds = _FakeMultiDataset(n_assets=3, n_days=120, lookback=30)
    rets = fib_analysis.next_day_returns(ds)
    assert rets.shape == (len(ds), 3)
    # Window s ends at calendar index s + w - 1; return is to day s + w
    s = 10
    expected = ds.prices_aligned[s + ds.lookback] / ds.prices_aligned[s + ds.lookback - 1] - 1.0
    np.testing.assert_allclose(rets[s], expected, rtol=1e-6)
    assert np.isfinite(rets).all()  # dataset construction gives every window a next day


def test_conditional_returns_selection():
    mask = np.zeros((100, 2), dtype=bool)
    mask[::4, 0] = True  # asset 0 near a level every 4th day
    rets = np.arange(200, dtype=float).reshape(100, 2) * 0.001
    out = fib_analysis.conditional_returns(mask, rets)
    s = out[(0, 1)]
    np.testing.assert_allclose(s["cond"], rets[::4, 1])
    assert s["n_cond"] == 25
    assert np.isfinite(s["ks_stat"]) and 0 <= s["ks_p"] <= 1
    assert np.isnan(out[(0, 0)]["ks_stat"])  # diagonal is not a cross-asset test


def test_fsm_events_per_asset_structure():
    ds = _FakeMultiDataset(n_assets=3, n_days=220, lookback=40)
    counts = fib_analysis.fsm_events_per_asset(ds.prices_aligned)
    assert len(counts) == 3
    for c in counts:
        assert set(c) == {"Bounce", "Break", "Hover", "Timeout"}
        assert all(isinstance(v, int) and v >= 0 for v in c.values())


def test_fib_analysis_end_to_end(tmp_path):
    ds = _FakeMultiDataset(n_assets=3, n_days=220, lookback=40, rho=0.6, seed=3)
    out = fib_analysis.main_fib_analysis(
        dataset=ds, output_dir=tmp_path, eps=0.05, segment_days=60, max_lag=5
    )
    for name in [
        "figA_simultaneous_levels.png",
        "figB_distance_heatmap.png",
        "check_a_signed_distance_hist.png",
        "check_a_fsm_events.png",
        "check_b_cooccurrence_lift.png",
        "check_b_lagged_xcorr.png",
        "check_c_conditional_ks_heatmap.png",
        "check_c_conditional_overlays.png",
        "exp2_findings.md",
    ]:
        assert (tmp_path / name).exists(), name
    assert "cooccurrence" in out and "conditional" in out
    findings = (tmp_path / "exp2_findings.md").read_text()
    assert "asset0" in findings and "simultaneously" in findings.lower()


# ---------------------------------------------------------------------------
# Experiment 3: ablation flags (use_context / learn_corr)
# ---------------------------------------------------------------------------


def test_ablation_flags_construct_all_combos():
    for use_context in (True, False):
        for learn_corr in (True, False):
            model = NeuralSDEMoEMultiAsset(
                lookback_window=LOOKBACK, n_assets=3, hidden_sizes=[32, 16],
                use_context=use_context, learn_corr=learn_corr,
            )
            x = torch.randn(4, LOOKBACK, 3)
            f = torch.randn(4, 3, FIB_NUM_LEVELS) * 0.1
            mu, chol_sigma, pi = model(x, f)
            assert mu.shape == (4, 3)
            assert chol_sigma.shape == (4, 3, 3)
            assert pi.shape == (4, 3, 3)
            assert model.chol_corr_param.requires_grad is learn_corr


def test_learn_corr_false_keeps_identity_after_training():
    from lightning import Trainer

    syn = SyntheticCorrelatedGBMDataset(n_assets=3, lookback_window=LOOKBACK, num_paths=16, steps_per_path=8)
    runner = _make_runner(n_assets=3, corr_init=syn.asset_corr_init, learn_corr=False)
    loader = DataLoader(syn, batch_size=32)
    trainer = Trainer(
        max_epochs=1,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        accelerator="cpu",
    )
    trainer.fit(runner, train_dataloaders=loader, val_dataloaders=loader)
    assert torch.isfinite(runner.trainer.callback_metrics["val/loss"])
    # Frozen at identity despite a nonzero data-driven corr_init being passed
    assert torch.allclose(runner.model.chol_corr_param.data, torch.eye(3))
    assert torch.allclose(runner.model.correlation_matrix(), torch.eye(3), atol=1e-6)


def test_use_context_false_blocks_cross_asset_information():
    torch.manual_seed(0)
    n = 3
    model_off = NeuralSDEMoEMultiAsset(lookback_window=LOOKBACK, n_assets=n, hidden_sizes=[32, 16], use_context=False)
    model_on = NeuralSDEMoEMultiAsset(lookback_window=LOOKBACK, n_assets=n, hidden_sizes=[32, 16], use_context=True)
    model_on.load_state_dict(model_off.state_dict())  # identical weights
    model_off.eval()
    model_on.eval()

    x = torch.randn(5, LOOKBACK, n)
    f = torch.randn(5, n, FIB_NUM_LEVELS) * 0.1
    x2 = x.clone()
    x2[:, :, 1] = torch.flip(x[:, :, 1], dims=[0])  # scramble asset 1's history

    with torch.no_grad():
        mu_off_a, _, pi_off_a = model_off(x, f)
        mu_off_b, _, pi_off_b = model_off(x2, f)
        mu_on_a, _, _ = model_on(x, f)
        mu_on_b, _, _ = model_on(x2, f)

    others = [0, 2]
    # Without context, other assets' outputs are invariant to asset 1's history
    assert torch.allclose(mu_off_a[:, others], mu_off_b[:, others], atol=1e-6)
    assert torch.allclose(pi_off_a[:, others], pi_off_b[:, others], atol=1e-6)
    # With context, they change
    assert not torch.allclose(mu_on_a[:, 0], mu_on_b[:, 0])


def test_nll_with_identity_corr_equals_sum_of_univariate():
    runner = _make_runner(n_assets=3, corr_init=None, learn_corr=False)
    model = runner.model
    x = torch.randn(7, LOOKBACK, 3)
    f = torch.randn(7, 3, FIB_NUM_LEVELS) * 0.1
    mu, chol_sigma, _ = model(x, f)

    off_diag = ~torch.eye(3, dtype=bool)
    assert torch.allclose(chol_sigma[:, off_diag], torch.zeros(7, 6))  # R = I -> diagonal

    x_t = x[:, -1, :]
    x_tp1 = x_t + torch.randn_like(x_t) * 0.01
    nll = runner._step_nll(x_t, x_tp1, mu, chol_sigma)

    sigma = chol_sigma.diagonal(dim1=-2, dim2=-1)
    dt = runner.default_dt
    dx = x_tp1 - x_t - mu * dt
    var = sigma**2 * dt
    manual = (0.5 * (math.log(2.0 * math.pi) + torch.log(var) + dx**2 / var)).sum(-1).mean()
    assert torch.allclose(nll, manual, atol=1e-3)  # runner adds an eps diagonal jitter


def test_exp3_ablation_cfg_modules_build():
    import importlib.util

    cfg_dir = Path(__file__).resolve().parents[1] / "experiments" / "neural_SDE" / "trainer_cfg" / "neural_SDE"
    expected = {
        "exp3_independent": (False, False),
        "exp3_corr_only": (False, True),
        "exp3_context_only": (True, False),
        "exp3_full": (True, True),
    }
    for name, (use_context, learn_corr) in expected.items():
        spec = importlib.util.spec_from_file_location(name, cfg_dir / f"{name}.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        cfg = module.get_trainer_cfg()
        assert cfg["model_cfg"]["_target_"] is NeuralSDEMoEMultiAsset
        assert cfg["model_cfg"]["use_context"] is use_context
        assert cfg["model_cfg"]["learn_corr"] is learn_corr
        assert cfg["dset_cfg"]["n_assets"] == 5
        assert cfg["run_cfg"]["rng_seed"] == 42
        assert cfg["run_cfg"]["run_name"] == name
