from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

ORIGINAL_DATA = ROOT / "workspace" / "portfolio_analysis" / "data"
EXTENSION_DATA = ROOT / "workspace" / "multiasset_extensions" / "data"
OUTPUT_DIR = ROOT / "workspace" / "multiasset_extensions" / "relationships"

PORTFOLIO_FILES = {
    "diversified": ORIGINAL_DATA / "diversified_prices.csv",
    "concentrated": ORIGINAL_DATA / "concentrated_prices.csv",
    "ultra_concentrated": EXTENSION_DATA / "ultra_concentrated_prices.csv"
}

PERIODS = {
    "2016-2017": ("2016-01-01", "2017-12-31"),
    "2018-2019": ("2018-01-01", "2019-12-31"),
    "2020": ("2020-01-01", "2020-12-31"),
    "full_2015-2020": ("2015-01-01", "2020-12-31")
}

ROLLING_WINDOW = 60
LAGS = range(-5, 6)

def load_portfolios():

    portfolios = {}

    for name, path in PORTFOLIO_FILES.items():

        if not path.exists():
            raise FileNotFoundError(path)

        prices = pd.read_csv(path, parse_dates=["Date"]).set_index("Date").sort_index()

        prices.columns = prices.columns.astype(str)

        if prices.index.has_duplicates:
            raise ValueError(f"Duplicate dates: {name}")

        if prices.shape[1] != 4:
            raise ValueError(f"Expected four assets: {name}")

        if (
            prices.isna().any().any()
            or not np.isfinite(prices.to_numpy()).all()
            or (prices <= 0).any().any()
        ):
            raise ValueError(f"Invalid prices: {name}")

        portfolios[name] = prices

    reference = portfolios["diversified"].index
    for name, prices in portfolios.items():
        if not prices.index.equals(reference):
            raise ValueError(f"Trading calendar differs: {name}")

    print("Portfolio price validation: PASSED")
    print(f"Price dates: {len(reference):,}")
    print(f"Period: {reference.min().date()} to {reference.max().date()}")

    return portfolios


def calculate_rolling_mean(returns):

    pairwise_rolling = []

    for a, b in combinations(returns.columns, 2):

        values = returns[a].rolling(ROLLING_WINDOW, min_periods=ROLLING_WINDOW).corr(returns[b])
        pairwise_rolling.append(values)

    matrix = pd.concat(pairwise_rolling, axis=1)
    return matrix.mean(axis=1)


def analyze_portfolios(portfolios):

    summary_rows = []
    pair_rows = []
    lag_rows = []
    rolling_rows = []

    for name, prices in portfolios.items():

        returns = prices.pct_change(fill_method=None)
        rolling_mean = calculate_rolling_mean(returns)

        for date, value in rolling_mean.dropna().items():

            rolling_rows.append({
                "Date": date,
                "Portfolio": name,
                "RollingMeanPairCorrelation": float(value)
            })

        for period, (start, end) in PERIODS.items():

            sample = returns.loc[start:end].dropna(how="any")

            if len(sample) < 30:
                raise ValueError(f"Insufficient returns: {name} / {period}")

            corr = sample.corr()
            cov = sample.cov()

            if (
                not np.isfinite(corr.to_numpy()).all()
                or not np.isfinite(cov.to_numpy()).all()
            ):
                raise ValueError(f"Invalid matrices: {name} / {period}")

            eigenvalues = np.linalg.eigvalsh(corr.to_numpy())
            pc1_share = (float(eigenvalues[-1] / eigenvalues.sum()))

            corr_values = []
            zero_lag_peak_count = 0

            for a, b in combinations(sample.columns, 2):

                pair_corr = float(corr.loc[a, b])
                pair_cov = float(cov.loc[a, b])
                same_direction = float((np.sign(sample[a]) == np.sign(sample[b])).mean())

                corr_values.append(pair_corr)
                pair_rows.append({
                    "Portfolio": name,
                    "Period": period,
                    "IssueId1": a,
                    "IssueId2": b,
                    "N_ReturnDays": len(sample),
                    "Correlation": pair_corr,
                    "Covariance": pair_cov,
                    "SameDaySignAgreement": same_direction
                })

                lag_values = {}
                for lag in LAGS:
                    value = float(sample[a].corr(sample[b].shift(lag)))
                    lag_values[lag] = value
                    lag_rows.append({
                        "Portfolio": name,
                        "Period": period,
                        "IssueId1": a,
                        "IssueId2": b,
                        "LagDays": lag,
                        "Correlation": value
                    })

                if not np.isfinite(list(lag_values.values())).all():
                    raise ValueError(f"Invalid lag correlations: {name} / {period} / {a} / {b}")

                best_lag = max(lag_values, key=lambda lag: abs(lag_values[lag]))

                if best_lag == 0:
                    zero_lag_peak_count += 1

            period_rolling = rolling_mean.loc[start:end].dropna()

            if period_rolling.empty:
                raise ValueError(f"No rolling correlations: {name} / {period}")

            summary_rows.append({
                "Portfolio": name,
                "Period": period,
                "N_ReturnDays": len(sample),
                "MeanPairCorrelation": float(np.mean(corr_values)),
                "MinPairCorrelation": float(np.min(corr_values)),
                "MaxPairCorrelation": float(np.max(corr_values)),
                "PC1ExplainedVariancePct": 100 * pc1_share,
                "MeanRolling60dCorrelation": float(period_rolling.mean()),
                "MedianRolling60dCorrelation": float(period_rolling.median()),
                "ZeroLagStrongestPairs": zero_lag_peak_count,
                "TotalPairs": len(corr_values)
            })

    return pd.DataFrame(summary_rows), pd.DataFrame(pair_rows), pd.DataFrame(lag_rows), pd.DataFrame(rolling_rows)


def validate_previous_results(summary):

    path = EXTENSION_DATA / "homebuilder_period_correlation_preview.csv"

    if not path.exists():
        raise FileNotFoundError(path)

    previous = pd.read_csv(path)

    columns = [
        "N_ReturnDays",
        "MeanPairCorrelation",
        "MinPairCorrelation",
        "MaxPairCorrelation"
    ]

    comparison = summary.merge(
        previous[["Portfolio", "Period"] + columns], 
        on=["Portfolio", "Period"], suffixes=("", "_Previous"), validate="one_to_one")

    if len(comparison) != 12:
        raise ValueError("Expected 12 matching portfolio-period summaries")

    for column in columns:
        if not np.allclose(comparison[column].to_numpy(), comparison[f"{column}_Previous"].to_numpy(), atol=1e-10, rtol=0):
            raise ValueError(f"Previous result mismatch: {column}")

    print("Previous correlation results reproduced: PASSED")


def main():

    print("=" * 90)
    print("MULTI-ASSET HISTORICAL RELATIONSHIP ANALYSIS BY PERIOD")
    print("=" * 90)

    portfolios = load_portfolios()
    summary, pairs, lags, rolling = analyze_portfolios(portfolios)
    validate_previous_results(summary)

    print("\nPORTFOLIO-PERIOD SUMMARY")
    display_columns = [
        "Portfolio",
        "Period",
        "N_ReturnDays",
        "MeanPairCorrelation",
        "MinPairCorrelation",
        "PC1ExplainedVariancePct",
        "MeanRolling60dCorrelation",
        "ZeroLagStrongestPairs"
    ]
    print(summary[display_columns].to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUTPUT_DIR / "relationship_period_summary.csv", index=False)
    pairs.to_csv(OUTPUT_DIR / "relationship_period_pairwise.csv", index=False)
    lags.to_csv(OUTPUT_DIR / "relationship_period_lagged.csv", index=False)
    rolling.to_csv(OUTPUT_DIR / "relationship_rolling_60d.csv", index=False)

    print("\nOUTPUT ROW COUNTS")
    print(f"Portfolio-period summary : {len(summary):,}")
    print(f"Pairwise measurements    : {len(pairs):,}")
    print(f"Lag correlations         : {len(lags):,}")
    print(f"Rolling daily values     : {len(rolling):,}")

    print("\nSaved results under:")
    print(OUTPUT_DIR)

    print("\nHISTORICAL RELATIONSHIP ANALYSIS COMPLETE")


if __name__ == "__main__":
    main()