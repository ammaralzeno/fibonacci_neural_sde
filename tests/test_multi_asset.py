"""Tests for the multi-asset data pipeline (Phase 1) and model (Phase 2)."""

import math

import numpy as np
import pandas as pd
import torch
from torch import nn

from amgm.data.neural_SDE_multi import (
    FIB_NUM_LEVELS,
    MultiAssetNeuralSDEDataset,
    SyntheticCorrelatedGBMDataset,
)
from amgm.models.mlp_multi import NeuralSDEMoEMultiAsset
import amgm.utils.common as common

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


def _make_runner(n_assets=4, corr_init=None):
    model_cfg = dict(
        _target_=NeuralSDEMoEMultiAsset,
        lookback_window=LOOKBACK,
        n_assets=n_assets,
        num_features=FIB_NUM_LEVELS,
        hidden_sizes=[32, 16],
        corr_init=corr_init,
    )
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
