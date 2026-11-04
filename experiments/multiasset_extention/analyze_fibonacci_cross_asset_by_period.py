from itertools import permutations
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

ORIGINAL_PRICES = ROOT / "workspace" / "portfolio_analysis" / "data"
EXTENSION_DATA = ROOT / "workspace" / "multiasset_extensions" / "data"
FIB_DIR = ROOT / "workspace" / "multiasset_extensions" / "fibonacci"
ORIGINAL_FIB = ROOT / "workspace" / "fibonacci_multiasset" / "data"

PRICE_FILES = {
    "diversified": ORIGINAL_PRICES / "diversified_prices.csv",
    "concentrated": ORIGINAL_PRICES / "concentrated_prices.csv",
    "ultra_concentrated": EXTENSION_DATA / "ultra_concentrated_prices.csv"
}

EVENT_FILE = FIB_DIR / "portfolio_fibonacci_event_state.csv"
SUMMARY_FILE = FIB_DIR / "portfolio_fibonacci_cross_asset_by_period.csv"
DETAIL_FILE = FIB_DIR / "portfolio_fibonacci_cross_asset_event_responses.csv"
ORIGINAL_SUMMARY = ORIGINAL_FIB / "diversified_fibonacci_cross_asset_summary.csv"

PERIODS = {
    "2016-2017": ("2016-01-01", "2017-12-31"),
    "2018-2019": ("2018-01-01", "2019-12-31"),
    "2020": ("2020-01-01", "2020-12-31"),
    "full_2015-2020": ("2015-01-01", "2020-12-31")
}

HORIZONS = [1, 5, 20]

ASSET_NAMES = {
    "diversified": {
        "00138001": "Energy",
        "00107801": "Health Care",
        "00141402": "Financials",
        "00116101": "Information Technology"
    },
    "concentrated": {
        "00116101": "IT 1161",
        "00169001": "IT 1690",
        "00177301": "IT 1773",
        "00187801": "IT 1878"
    },
    "ultra_concentrated": {
        "00666901": "Lennar",
        "00882301": "PulteGroup",
        "01239501": "Toll Brothers",
        "01268901": "KB Home"
    }
}

def load_inputs():

    portfolios = {}

    for portfolio, path in PRICE_FILES.items():

        if not path.exists():
            raise FileNotFoundError(path)

        prices = pd.read_csv(path, parse_dates=["Date"]).set_index("Date").sort_index()
        prices.columns = prices.columns.astype(str)

        if (
            prices.shape[1] != 4
            or prices.index.has_duplicates
            or prices.columns.has_duplicates
        ):
            raise ValueError(f"Invalid price panel: {portfolio}")

        values = prices.to_numpy(dtype=float)

        if not np.isfinite(values).all() or (values <= 0).any():
            raise ValueError(f"Invalid prices: {portfolio}")

        if set(prices.columns) != set(ASSET_NAMES[portfolio]):
            raise ValueError(f"Unexpected asset membership: {portfolio}")

        portfolios[portfolio] = prices

    reference = portfolios["diversified"].index

    if len(reference) != 1512:
        raise ValueError("Unexpected price calendar length")

    for portfolio, prices in portfolios.items():

        if not prices.index.equals(reference):
            raise ValueError(f"Calendar mismatch: {portfolio}")

    if not EVENT_FILE.exists():
        raise FileNotFoundError(EVENT_FILE)

    events = pd.read_csv(EVENT_FILE, parse_dates=["Date"], dtype={"IssueId": str})

    if len(events) != 15132:
        raise ValueError("Unexpected event-state row count")

    if events.duplicated(["Portfolio", "IssueId", "Date"]).any():
        raise ValueError("Duplicate event-state records")

    flags = events["InteriorFibEventStart"].astype(str).str.lower().map({"true": True, "false": False})

    if flags.isna().any():
        raise ValueError("Invalid event-start flags")

    events["InteriorFibEventStart"] = flags.astype(bool)

    expected_dates = reference[251:]

    event_matrices = {}

    for portfolio, prices in portfolios.items():

        subset = events[events["Portfolio"] == portfolio]

        matrix = (subset.pivot(index="Date", columns="IssueId", values="InteriorFibEventStart")
            .reindex(index=expected_dates, columns=prices.columns))

        if len(subset) != 4 * len(expected_dates) or matrix.isna().any().any():
            raise ValueError(f"Unbalanced event panel: {portfolio}")

        event_matrices[portfolio] = matrix.astype(bool)

    print("Price and event alignment: PASSED")

    return portfolios, event_matrices


def statistics(values):

    values = pd.Series(values).dropna().astype(float)

    if values.empty:
        raise ValueError("Empty return-comparison sample")

    return {
        "N": len(values),
        "Mean": float(values.mean()),
        "Median": float(values.median()),
        "PositiveRate": float((values > 0).mean()),
        "MeanAbs": float(values.abs().mean())
    }


def sign_agreement(daily, dates, source, target):

    source_values = daily.loc[dates, source]
    target_values = daily.loc[dates, target]

    valid = source_values.notna() & target_values.notna()

    if not valid.any():
        raise ValueError("No valid sign-comparison observations")

    return float((np.sign(source_values.loc[valid]) == np.sign(target_values.loc[valid])).mean())


def make_summary_row(portfolio, period, source, target, horizon, event_stats, baseline_stats, event_sign=np.nan, baseline_sign=np.nan):

    if horizon == 0:
        window = "same_day"
    else:
        window = f"forward_{horizon}d"

    return {
        "Portfolio": portfolio,
        "Period": period,
        "SourceIssueId": source,
        "SourceAsset": ASSET_NAMES[portfolio][source],
        "TargetIssueId": target,
        "TargetAsset": ASSET_NAMES[portfolio][target],
        "ResponseWindow": window,
        "HorizonDays": horizon,
        "EventN": event_stats["N"],
        "BaselineN": baseline_stats["N"],
        "EventMeanReturn": event_stats["Mean"],
        "BaselineMeanReturn": baseline_stats["Mean"],
        "MeanReturnDifference": event_stats["Mean"] - baseline_stats["Mean"],
        "EventMedianReturn": event_stats["Median"],
        "BaselineMedianReturn": baseline_stats["Median"],
        "EventPositiveRate": event_stats["PositiveRate"],
        "BaselinePositiveRate": baseline_stats["PositiveRate"],
        "PositiveRateDifference": event_stats["PositiveRate"] - baseline_stats["PositiveRate"],
        "EventMeanAbsReturn": event_stats["MeanAbs"],
        "BaselineMeanAbsReturn": baseline_stats["MeanAbs"],
        "MeanAbsReturnDifference": event_stats["MeanAbs"] - baseline_stats["MeanAbs"],
        "EventSignAgreement": event_sign,
        "BaselineSignAgreement": baseline_sign,
        "SignAgreementDifference": event_sign - baseline_sign
    }


def analyze_portfolio(portfolio, prices, events):

    rows = []
    details = []

    daily = prices.pct_change(fill_method=None)

    for period, (start, end) in PERIODS.items():

        period_events = events.loc[start:end]
        period_prices = prices.loc[start:end]

        forwards = {horizon: (period_prices.shift(-horizon) / period_prices - 1) for horizon in HORIZONS}

        for source, target in permutations(prices.columns, 2):

            source_flags = period_events[source]

            event_dates = source_flags.index[source_flags]
            normal_dates = source_flags.index[~source_flags]

            if len(event_dates) == 0 or len(normal_dates) == 0:
                raise ValueError(f"Empty event or baseline sample: {portfolio}/{period}/{source}")

            event_sign = sign_agreement(daily, event_dates, source, target)
            baseline_sign = sign_agreement(daily, normal_dates, source, target)

            same_day = daily[target]

            event_stats = statistics(same_day.reindex(event_dates))
            baseline_stats = statistics(same_day.reindex(normal_dates))

            rows.append(
                make_summary_row(
                    portfolio,
                    period,
                    source,
                    target,
                    0,
                    event_stats,
                    baseline_stats,
                    event_sign,
                    baseline_sign
                )
            )

            for horizon in HORIZONS:

                target_forward = forwards[horizon][target]

                event_stats = statistics(target_forward.reindex(event_dates))
                baseline_stats = statistics(target_forward.reindex(normal_dates))

                rows.append(
                    make_summary_row(
                        portfolio,
                        period,
                        source,
                        target,
                        horizon,
                        event_stats,
                        baseline_stats
                    )
                )

            for date in event_dates:

                record = {
                    "Portfolio": portfolio,
                    "Period": period,
                    "Date": date,
                    "SourceIssueId": source,
                    "TargetIssueId": target,
                    "SourceSameDayReturn": daily.at[date, source],
                    "TargetSameDayReturn": daily.at[date, target]
                }

                for horizon in HORIZONS:
                    record[f"TargetForwardReturn_{horizon}d"] = forwards[horizon].at[date, target]

                details.append(record)

    return rows, details


def validate_original(results):

    if not ORIGINAL_SUMMARY.exists():
        raise FileNotFoundError(ORIGINAL_SUMMARY)

    original = pd.read_csv(ORIGINAL_SUMMARY, dtype={"SourceIssueId": str, "TargetIssueId": str})

    current = results[(results["Portfolio"] == "diversified") & (results["Period"] == "full_2015-2020")]
    keys = ["SourceIssueId", "TargetIssueId", "HorizonDays"]

    if len(original) != 48 or len(current) != 48:
        raise ValueError("Expected 48 original cross-asset rows")

    if (original.duplicated(keys).any() or current.duplicated(keys).any()):
        raise ValueError("Duplicate comparison keys")

    columns = [
        "EventN",
        "EventMeanReturn",
        "BaselineMeanReturn",
        "MeanReturnDifference",
        "EventPositiveRate",
        "PositiveRateDifference",
        "EventMeanAbsReturn",
        "BaselineMeanAbsReturn",
        "EventSignAgreement",
        "BaselineSignAgreement",
        "SignAgreementDifference"
    ]

    merged = current.merge(original[keys + columns], on=keys, suffixes=("", "_Original"), validate="one_to_one")

    if len(merged) != 48:
        raise ValueError("Original comparison is incomplete")

    for column in columns:

        if not np.allclose(merged[column].to_numpy(dtype=float), merged[f"{column}_Original"].to_numpy(dtype=float), rtol=0, atol=1e-10, equal_nan=True):
            raise ValueError(f"Original cross-asset result mismatch: {column}")

    print("Original diversified cross-asset results: REPRODUCED")


def print_comparisons(results):

    primary = results[results["Period"] != "full_2015-2020"].copy()

    same_day = primary[primary["HorizonDays"] == 0].copy()
    same_day["AbsReturnMultiplier"] = same_day["EventMeanAbsReturn"] / same_day["BaselineMeanAbsReturn"]
    same_day["SignDifferencePP"] = 100 * same_day["SignAgreementDifference"]

    same_day_summary = (
        same_day
        .groupby(["Portfolio", "Period"], as_index=False)
        .agg(
            MeanAbsReturnMultiplier=("AbsReturnMultiplier", "mean"),
            MeanSignDifferencePP=("SignDifferencePP", "mean"),
            MinEventN=("EventN", "min")
        )
    )

    forward = primary[primary["HorizonDays"] > 0].copy()
    forward["DifferencePP"] = 100 * forward["MeanReturnDifference"]

    forward_summary = (
        forward
        .groupby(["Portfolio", "Period", "HorizonDays"], as_index=False)
        .agg(
            MeanDifferencePP=("DifferencePP", "mean"),
            MinEventN=("EventN", "min")
        )
    )

    print("\nSAME-DAY CROSS-ASSET RESPONSE")
    print("(Equal-weight average across 12 directed pairs)")

    print(same_day_summary.to_string(index=False, float_format=lambda x: f"{x:+.3f}"))

    print("\nCROSS-ASSET FORWARD RETURN DIFFERENCES")
    print("(Percentage points; average across 12 directed pairs)")

    print(forward_summary.to_string(index=False, float_format=lambda x: f"{x:+.3f}"))


def main():

    print("=" * 90)
    print("FIBONACCI CROSS-ASSET RESPONSES BY PORTFOLIO AND PERIOD")
    print("=" * 90)

    portfolios, event_matrices = load_inputs()

    rows = []
    details = []

    for portfolio, prices in portfolios.items():

        summary_rows, detail_rows = analyze_portfolio(portfolio, prices, event_matrices[portfolio])
        rows.extend(summary_rows)
        details.extend(detail_rows)

    results = pd.DataFrame(rows)
    detail = pd.DataFrame(details)

    expected_rows = 3 * 4 * 12 * 4

    if len(results) != expected_rows:
        raise ValueError(f"Expected {expected_rows} summary rows, found {len(results)}")

    expected_detail = 0

    for matrix in event_matrices.values():
        for start, end in PERIODS.values():
            expected_detail += 3 * int(matrix.loc[start:end].to_numpy().sum())

    if len(detail) != expected_detail:
        raise ValueError(f"Expected {expected_detail} detail rows, found {len(detail)}")

    validate_original(results)
    print_comparisons(results)

    FIB_DIR.mkdir(parents=True, exist_ok=True)

    results.to_csv(SUMMARY_FILE, index=False)
    detail.to_csv(DETAIL_FILE, index=False)

    print(f"\nSummary records: {len(results):,}")
    print(f"Detail records: {len(detail):,}")

    print("\nSaved:")
    print(SUMMARY_FILE)
    print(DETAIL_FILE)

    print("\nCROSS-ASSET RESPONSE ANALYSIS COMPLETE")


if __name__ == "__main__":
    main()