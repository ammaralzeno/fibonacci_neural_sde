import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from amgm import config as amgm_config
from amgm.data.loading import load_sebx_am_data


SOURCE_DIR = ROOT / "workspace" / "portfolio_analysis"
OUTPUT_DIR = ROOT / "workspace" / "multiasset_extensions" / "data"

HOMEBUILDERS = {
    "00666901": ("Lennar", "LEN"),
    "00882301": ("PulteGroup", "PHM"),
    "01239501": ("Toll Brothers", "TOL"),
    "01268901": ("KB Home", "KBH")
}

IDS = list(HOMEBUILDERS)
SUBINDUSTRY = 25201030

PERIODS = {
    "2016-2017": ("2016-01-01", "2017-12-31"),
    "2018-2019": ("2018-01-01", "2019-12-31"),
    "2020": ("2020-01-01", "2020-12-31"),
    "full_2015-2020": ("2015-01-01", "2020-12-31")
}


def read_existing_prices(name):

    path = SOURCE_DIR / "data" / f"{name}_prices.csv"
    prices = pd.read_csv(path, parse_dates=["Date"]).set_index("Date").sort_index()

    prices.columns = prices.columns.astype(str)
    if prices.index.has_duplicates:
        raise ValueError(f"Duplicate dates in {name}")

    if (
        prices.shape[1] != 4
        or prices.isna().any().any()
        or (prices <= 0).any().any()
    ):
        raise ValueError(f"Invalid {name} portfolio prices")

    return prices


def validate_metadata():

    metadata = pd.read_csv(SOURCE_DIR / "phase1" / "constituents_metadata_preview.csv", dtype={"IssueId": str})
    coverage = pd.read_csv(SOURCE_DIR / "phase1" / "issue_coverage.csv", dtype={"IssueId": str})

    if (
        metadata["IssueId"].duplicated().any()
        or coverage["IssueId"].duplicated().any()
    ):
        raise ValueError("Duplicate IssueIds in metadata")


    metadata = metadata.set_index("IssueId")
    coverage = coverage.set_index("IssueId")

    missing = set(IDS) - (set(metadata.index) & set(coverage.index))

    if missing:
        raise ValueError(f"Homebuilders missing from metadata/coverage: {sorted(missing)}")

    selected = metadata.loc[IDS].copy()
    history = coverage.loc[IDS]

    if not selected["Country"].eq("US").all():
        raise ValueError("Not all securities are US-listed")

    if not selected["SectorCode"].astype(int).eq(25).all():
        raise ValueError("Sector-code mismatch")

    if not selected["IndustryCode"].astype(int).eq(252010).all():
        raise ValueError("Industry-code mismatch")

    if not selected["SubIndustryCode"].astype(int).eq(SUBINDUSTRY).all():
        raise ValueError("Homebuilding sub-industry mismatch")

    full_history = (
        history["first_date"].eq("2014-12-31")
        & history["last_date"].eq("2020-12-31")
        & history["valid_prices"].ge(1512)
        & history["missing_prices"].eq(0)
    )

    if not full_history.all():
        raise ValueError("At least one homebuilder lacks the required history:\n" + history.loc[~full_history].to_string())

    print("Homebuilder metadata and history: PASSED")

    return selected


def load_homebuilder_prices(reference_dates):

    dataset = load_sebx_am_data(amgm_config.am_dataset_dir)

    security = dataset["security_data"]
    constituents = dataset["constituents"]

    members = constituents[["IssueId", "Gvkey"]].copy()
    members["IssueId"] = members["IssueId"].astype(str)

    if members["IssueId"].duplicated().any():
        raise ValueError("Duplicate IssueIds in constituents")

    company_gvkeys = members.set_index("IssueId")["Gvkey"].reindex(IDS)

    if company_gvkeys.isna().any():
        raise ValueError("Missing company GVKEY identifiers")

    if company_gvkeys.nunique() != 4:
        raise ValueError("The four securities are not distinct companies")

    for iid in IDS:
        expected = int(iid[:-2])
        actual = int(company_gvkeys.loc[iid])

        if actual != expected:
            raise ValueError(f"GVKEY mismatch for {iid}: expected {expected}, found {actual}")

    security = security.loc[security["IssueId"].astype(str).isin(IDS), ["IssueId", "Date", "ClAdjLoc", "CcyLoc"]].copy()

    security["IssueId"] = security["IssueId"].astype(str)
    security["Date"] = pd.to_datetime(security["Date"])

    if security.duplicated(["IssueId", "Date"]).any():
        raise ValueError("Duplicate historical price records")

    if not security["CcyLoc"].eq("USD").all():
        raise ValueError("Non-USD price history detected")

    prices = security.pivot(index="Date", columns="IssueId", values="ClAdjLoc"
                            ).reindex(index=reference_dates, columns=IDS).apply(pd.to_numeric, errors="coerce")

    if (
        prices.isna().any().any()
        or not np.isfinite(prices.to_numpy()).all()
        or (prices <= 0).any().any()
    ):
        raise ValueError("Missing or invalid aligned homebuilder prices")

    print("Homebuilder price alignment: PASSED")
    print(f"Price observations: {len(prices):,}")
    print(f"Period: {prices.index.min().date()} to {prices.index.max().date()}")

    return prices


def analyze_correlations(portfolios):

    summary_rows = []
    pair_rows = []

    for name, prices in portfolios.items():

        returns = prices.pct_change(fill_method=None)

        for period, (start, end) in PERIODS.items():

            sample = returns.loc[start:end].dropna(how="any")
            if len(sample) < 30:
                raise ValueError(f"Insufficient observations for {name}/{period}")

            correlation = sample.corr()
            pairs = [(a, b, float(correlation.loc[a, b])) for a, b in combinations(correlation.columns, 2)]
            values = np.array([value for _, _, value in pairs])

            if not np.isfinite(values).all():
                raise ValueError(f"Invalid correlations for {name}/{period}")

            summary_rows.append({
                "Portfolio": name,
                "Period": period,
                "N_ReturnDays": len(sample),
                "MeanPairCorrelation": float(values.mean()),
                "MinPairCorrelation": float(values.min()),
                "MaxPairCorrelation": float(values.max())
            })

            for a, b, value in pairs:
                pair_rows.append({
                    "Portfolio": name,
                    "Period": period,
                    "IssueId1": a,
                    "IssueId2": b,
                    "Correlation": value
                })

    return pd.DataFrame(summary_rows), pd.DataFrame(pair_rows)


def build_definitions(metadata, homebuilder_prices):

    original = pd.read_csv(SOURCE_DIR / "data" / "portfolio_definitions.csv", dtype={"IssueId": str})

    if original.groupby("portfolio")["IssueId"].nunique().to_dict() != {"diversified": 4, "concentrated": 4}:
        raise ValueError("Original portfolio definitions have changed")

    new_rows = []

    for iid in IDS:

        name, ticker = HOMEBUILDERS[iid]
        row = metadata.loc[iid]

        new_rows.append({
            "portfolio": "ultra_concentrated",
            "IssueId": iid,
            "Company": name,
            "Ticker": ticker,
            "Country": row["Country"],
            "SectorCode": int(row["SectorCode"]),
            "Sector": "Consumer Discretionary",
            "IndustryCode": int(row["IndustryCode"]),
            "SubIndustryCode": int(row["SubIndustryCode"]),
            "first_date": str(homebuilder_prices.index.min().date()),
            "last_date": str(homebuilder_prices.index.max().date()),
            "valid_prices": len(homebuilder_prices),
            "missing_prices": 0,
            "duplicate_dates": 0
        })

    combined = pd.concat([original, pd.DataFrame(new_rows)], ignore_index=True)

    if combined.groupby("portfolio")["IssueId"].nunique().to_dict() != {"diversified": 4, "concentrated": 4, "ultra_concentrated": 4}:
        raise ValueError("Unexpected combined portfolio membership")

    return combined


def main():

    print("=" * 90)
    print("HOMEBUILDER PORTFOLIO PREPARATION")
    print("=" * 90)

    diversified = read_existing_prices("diversified")
    concentrated = read_existing_prices("concentrated")

    if not diversified.index.equals(concentrated.index):
        raise ValueError("Original portfolios have different calendars")

    metadata = validate_metadata()
    homebuilders = load_homebuilder_prices(diversified.index)
    definitions = build_definitions(metadata, homebuilders)

    summary, pairs = analyze_correlations({
        "diversified": diversified,
        "concentrated": concentrated,
        "ultra_concentrated": homebuilders
    })

    print("\nHOMEBUILDER SECURITIES")
    for iid in IDS:
        company, ticker = HOMEBUILDERS[iid]
        print(f"  {iid}  {company} ({ticker})")

    print("\nPORTFOLIO / PERIOD CORRELATIONS")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    print("\nHOMEBUILDER PAIRWISE CORRELATIONS")
    homebuilder_pairs = pairs[pairs["Portfolio"] == "ultra_concentrated"]
    print(homebuilder_pairs.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    calibration_stats = summary[(summary["Portfolio"] == "ultra_concentrated") & (summary["Period"] == "2016-2017")].iloc[0]

    if calibration_stats["MinPairCorrelation"] < 0.70:
        print("\nWARNING: Not every homebuilder pair is strongly correlated during 2016-2017.")
        
    else:
        print("\nHomebuilder portfolio meets the initial 0.70 minimum-pair-correlation criterion.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    homebuilders.rename_axis("Date").to_csv(OUTPUT_DIR / "ultra_concentrated_prices.csv")
    definitions.to_csv(OUTPUT_DIR / "homebuilder_portfolio_definitions.csv", index=False)
    summary.to_csv(OUTPUT_DIR / "homebuilder_period_correlation_preview.csv", index=False)
    pairs.to_csv(OUTPUT_DIR / "homebuilder_pairwise_correlations.csv", index=False)

    print("\nSaved four files under:")
    print(OUTPUT_DIR)
    print("\nHOMEBUILDER PORTFOLIO PREPARATION COMPLETE")


if __name__ == "__main__":
    main()