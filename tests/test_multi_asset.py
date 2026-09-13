"""Phase 1 tests for the multi-asset data pipeline."""

import numpy as np
import pandas as pd
import torch

from amgm.data.neural_SDE_multi import (
    FIB_NUM_LEVELS,
    MultiAssetNeuralSDEDataset,
    SyntheticCorrelatedGBMDataset,
)
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
