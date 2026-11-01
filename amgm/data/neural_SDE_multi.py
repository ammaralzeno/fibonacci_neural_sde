"""Multi-asset datasets for the joint (portfolio) Neural SDE."""

from typing import Any, NamedTuple, Optional

import numpy as np
import pandas as pd
import polars as pl
import torch
from torch.utils.data import Dataset

from amgm import config as amgm_config
from amgm.data.base import BaseAMData
from amgm.data.loading import load_sebx_am_data
import amgm.utils.common as common

FIB_NUM_LEVELS = 7


class MultiAssetNeuralSDESample(NamedTuple):
    """price_window (w, N), nxt_price/sample_min/sample_range (N,), features/fib_levels (N, 7)."""

    price_window: torch.Tensor
    nxt_price: torch.Tensor
    sample_min: torch.Tensor
    sample_range: torch.Tensor
    features: torch.Tensor
    fib_levels: torch.Tensor
    test_dates: Any
    issue_ids: Any


def _min_max_normalize_multi(price_windows: np.ndarray, nxt_prices: np.ndarray):
    """Per-asset min-max normalization for (B, w, N) windows and (B, N) next prices."""
    pw_min = price_windows.min(axis=1, keepdims=True)
    pw_max = price_windows.max(axis=1, keepdims=True)
    pw_range = np.maximum(pw_max - pw_min, 1e-8)
    norm_windows = (price_windows - pw_min) / pw_range
    norm_nxt = (nxt_prices - pw_min[:, 0, :]) / pw_range[:, 0, :]
    return (
        norm_windows.astype(np.float32),
        norm_nxt.astype(np.float32),
        pw_min[:, 0, :].astype(np.float32),
        pw_range[:, 0, :].astype(np.float32),
    )


def _fib_features_multi(norm_windows: np.ndarray):
    """Per-asset Fibonacci levels and normalized signed distances for (B, w, N) windows."""
    B, w, N = norm_windows.shape
    flat = np.transpose(norm_windows, (0, 2, 1)).reshape(B * N, w)
    fib_levels, delta = common.calculate_fibLevels(flat)
    last_price = flat[:, -1:]
    features = (last_price - fib_levels) / np.maximum(delta, 1e-8)
    return (
        features.reshape(B, N, FIB_NUM_LEVELS).astype(np.float32),
        fib_levels.reshape(B, N, FIB_NUM_LEVELS).astype(np.float32),
    )


def _increment_correlation(norm_windows: np.ndarray, norm_nxt: np.ndarray) -> np.ndarray:
    """Correlation of normalized one-step increments dx = x_{t+1} - x_t; the natural init for the model's R."""
    dx = norm_nxt - norm_windows[:, -1, :]
    corr = np.corrcoef(dx, rowvar=False)
    corr = np.nan_to_num(corr, nan=0.0)
    np.fill_diagonal(corr, 1.0)
    return corr.astype(np.float32)


class MultiAssetNeuralSDEDataset(BaseAMData, Dataset):
    """Date-aligned multi-asset (basket) dataset for the joint Neural SDE.

    Builds rolling windows over a fixed basket of N assets sharing one trading
    calendar (inner join on Date: dates missing any basket asset are dropped).
    """

    def __init__(
        self,
        issue_ids: Optional[list[str]],
        n_assets: int = 10,
        start_date: str = "2014-12-31",
        end_date: str = "2017-12-31",
        lookback_window: int = 252,
        max_windows: Optional[int] = None,
        dt: float = 1.0,
        price_value_col: str = "ClAdjLoc",
        training_data_type: str = "US_Stocks",
        data_source: str = "local",
        dset_path=amgm_config.am_dataset_dir,
        uploaded_df=None,
        rng_seed: Optional[int] = None,
        **_,
    ):
        """
        Args:
            issue_ids: Basket asset IDs. If None or empty, the n_assets assets with
                the most complete date coverage in [start_date, end_date] are selected
                (deterministic: coverage desc, then IssueId asc).
            n_assets: Basket size, only used when issue_ids is None or empty.
            max_windows: Optional cap on the number of windows; windows are then
                subsampled evenly spaced (deterministic).
        """
        self.lookback_window = int(lookback_window)
        self.max_windows = max_windows
        self.dt = float(dt)
        cls_name = self.__class__.__qualname__

        if issue_ids is None or len(issue_ids) == 0:
            issue_ids = self._select_basket(dset_path, n_assets, start_date, end_date)
        print(f"{cls_name}: basket of {len(issue_ids)} assets: {issue_ids}")

        BaseAMData.__init__(
            self,
            issue_ids=issue_ids,
            start_date=start_date,
            end_date=end_date,
            feature_names=[price_value_col],
            normalization={},
            log_name=cls_name,
            data_source=data_source,
            training_data_type=training_data_type,
            dset_path=dset_path,
            uploaded_df=uploaded_df,
        )

        prices, dates, basket = self._pivot_aligned(self.securities, issue_ids, price_value_col)
        self.basket_issue_ids = basket
        self.n_assets = len(basket)
        self.dates = dates
        self.prices_aligned = prices  # (T, N) aligned price matrix, kept for fetching real futures during generation

        price_windows, nxt_prices, test_dates = self._make_windows(prices, dates)
        norm_windows, norm_nxt, sample_min, sample_range = _min_max_normalize_multi(price_windows, nxt_prices)
        features, fib_levels = _fib_features_multi(norm_windows)

        self.asset_corr_init = _increment_correlation(norm_windows, norm_nxt)

        print(
            f"{cls_name}: windows {norm_windows.shape}, next {norm_nxt.shape}, "
            f"features {features.shape}, aligned dates {len(dates)}"
        )
        assert np.isfinite(norm_windows).all() and np.isfinite(features).all()

        self.price_window = torch.from_numpy(norm_windows)
        self.nxt_prices = torch.from_numpy(norm_nxt)
        self.sample_min = torch.from_numpy(sample_min)
        self.sample_range = torch.from_numpy(sample_range)
        self.features = torch.from_numpy(features)
        self.fib_levels = torch.from_numpy(fib_levels)
        self.test_dates = test_dates

    @staticmethod
    def _select_basket(dset_path, n_assets: int, start_date: str, end_date: str) -> list[str]:
        """Pick the n_assets IssueIds with the most complete coverage in the date range."""
        data = load_sebx_am_data(dset_path)
        sd = data["security_data"]
        mask = (sd["Date"] >= pd.Timestamp(start_date)) & (sd["Date"] <= pd.Timestamp(end_date))
        counts = sd.loc[mask].groupby("IssueId")["Date"].count().sort_values(ascending=False)
        top = counts.iloc[:n_assets]
        basket = sorted(top.index.tolist())
        print(f"Basket coverage (rows per asset): {top.to_dict()}")
        return basket

    def _pivot_aligned(self, secs: pl.DataFrame, basket: list[str], price_value_col: str):
        """Pivot to a (T, N) price matrix on the shared calendar (inner join on Date)."""
        wide = secs.pivot(
            values=price_value_col, index="Date", on="IssueId", aggregate_function="first"
        ).sort("Date")
        available = [c for c in basket if c in wide.columns]
        dropped = [c for c in basket if c not in wide.columns]
        if dropped:
            self.log.warning(f"Assets with no data in range, excluded from basket: {dropped}")
        if len(available) < 2:
            raise ValueError(f"Need at least 2 assets with data, got {len(available)}: {available}")

        wide = wide.select(["Date"] + available)
        n_before = wide.height
        wide = wide.drop_nulls()
        n_dropped = n_before - wide.height
        if n_dropped > 0:
            print(f"Inner join dropped {n_dropped}/{n_before} dates with missing assets.")

        dates = wide["Date"].dt.strftime("%Y-%m-%d").to_list()
        prices = wide.select(available).to_numpy().astype(np.float32)
        return prices, dates, available

    def _make_windows(self, prices: np.ndarray, dates: list[str]):
        """Slide (w, N) windows over the aligned price matrix; target is the next day."""
        T = prices.shape[0]
        w = self.lookback_window
        num_windows = T - w
        if num_windows <= 0:
            raise ValueError(f"Not enough aligned dates ({T}) for lookback_window={w}.")

        starts = np.arange(num_windows)
        if self.max_windows is not None and num_windows > self.max_windows:
            starts = np.unique(np.linspace(0, num_windows - 1, self.max_windows).round().astype(int))

        price_windows = np.stack([prices[s : s + w] for s in starts])
        nxt_prices = np.stack([prices[s + w] for s in starts])
        test_dates = [dates[s + w - 1] for s in starts]
        return price_windows, nxt_prices, test_dates

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return MultiAssetNeuralSDESample(
            price_window=self.price_window[idx],
            nxt_price=self.nxt_prices[idx],
            sample_min=self.sample_min[idx],
            sample_range=self.sample_range[idx],
            features=self.features[idx],
            fib_levels=self.fib_levels[idx],
            test_dates=self.test_dates[idx],
            issue_ids=self.basket_issue_ids,
        )


class SyntheticCorrelatedGBMDataset(Dataset):
    """Synthetic N-asset correlated GBM with known ground-truth correlation.

    dX_i = mu_i X_i dt + sigma_i X_i (L_R dW)_i, with equicorrelation
    R = (1 - rho) I + rho 11'. Used to verify that the multivariate NLL and the
    model's correlation module can recover a known correlation matrix before
    training on real data.
    """

    def __init__(
        self,
        n_assets: int = 5,
        lookback_window: int = 252,
        dt: float = 1.0 / 252.0,
        num_paths: int = 64,
        steps_per_path: int = 32,
        s0: float = 100.0,
        mu_low: float = -0.05,
        mu_high: float = 0.15,
        sigma_low: float = 0.10,
        sigma_high: float = 0.40,
        rho: float = 0.40,
        rng_seed: int = 1,
        **_,
    ):
        self.n_assets = int(n_assets)
        self.lookback_window = int(lookback_window)
        self.dt = float(dt)

        gen = np.random.default_rng(int(rng_seed))
        N = self.n_assets
        self.true_corr = ((1.0 - rho) * np.eye(N) + rho * np.ones((N, N))).astype(np.float32)
        self.true_mu = gen.uniform(mu_low, mu_high, size=N).astype(np.float32)
        self.true_sigma = gen.uniform(sigma_low, sigma_high, size=N).astype(np.float32)
        self.basket_issue_ids = [f"asset{i}" for i in range(N)]

        chol_corr = np.linalg.cholesky(self.true_corr)
        drift = (self.true_mu - 0.5 * self.true_sigma**2) * self.dt
        vol_sqrt_dt = self.true_sigma * np.sqrt(self.dt)

        price_windows, nxt_prices, test_dates = [], [], []
        w = self.lookback_window
        for path_idx in range(int(num_paths)):
            n_total = w + int(steps_per_path)
            z = gen.standard_normal((n_total, N))
            shocks = z @ chol_corr.T * vol_sqrt_dt  # (n_total, N) correlated log-returns
            log_x = np.log(s0) + np.cumsum(drift + shocks, axis=0)
            x = np.exp(log_x).astype(np.float32)  # (n_total, N)

            for start in range(int(steps_per_path)):
                price_windows.append(x[start : start + w])
                nxt_prices.append(x[start + w])
                test_dates.append(f"synthetic_path{path_idx}_step{start}")

        price_windows = np.stack(price_windows)
        nxt_prices = np.stack(nxt_prices)
        norm_windows, norm_nxt, sample_min, sample_range = _min_max_normalize_multi(price_windows, nxt_prices)
        features, fib_levels = _fib_features_multi(norm_windows)
        self.asset_corr_init = _increment_correlation(norm_windows, norm_nxt)

        self.price_window = torch.from_numpy(norm_windows)
        self.nxt_prices = torch.from_numpy(norm_nxt)
        self.sample_min = torch.from_numpy(sample_min)
        self.sample_range = torch.from_numpy(sample_range)
        self.features = torch.from_numpy(features)
        self.fib_levels = torch.from_numpy(fib_levels)
        self.test_dates = test_dates

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return MultiAssetNeuralSDESample(
            price_window=self.price_window[idx],
            nxt_price=self.nxt_prices[idx],
            sample_min=self.sample_min[idx],
            sample_range=self.sample_range[idx],
            features=self.features[idx],
            fib_levels=self.fib_levels[idx],
            test_dates=self.test_dates[idx],
            issue_ids=self.basket_issue_ids,
        )
