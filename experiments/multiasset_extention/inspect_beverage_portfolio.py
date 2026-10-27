from __future__ import annotations

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
PRICES_DIR = SOURCE_DIR / "data"
OUTPUT_DIR = ROOT / "workspace" / "multiasset_extensions" / "data"

SUBINDUSTRY = 30201030

BRANDS = {
    "00314401": (3144, "The Coca-Cola Company", "KO"),
    "00847901": (8479, "PepsiCo", "PEP"),
    "02431601": (24316, "Monster Beverage", "MNST"),
    "02446301": (24463, "National Beverage", "FIZZ"),
    "00313801": (3138, "Coca-Cola Consolidated", "COKE"),
    "02033801": (20338, "Cott / Primo Water", "COT / PRMW")
}
ANCHORS = {"00314401", "00847901"}
PORTFOLIO_SIZE = 4
CALIBRATION_START = "2016-01-01"
CALIBRATION_END = "2017-12-31"

PERIODS = {
    "2016-2017_selection": ("2016-01-01", "2017-12-31"),
    "2018-2019_evaluation": ("2018-01-01", "2019-12-31"),
    "2020_evaluation": ("2020-01-01", "2020-12-31"),
    "full_2015-2020": ("2015-01-01", "2020-12-31")
}

def read_existing_prices(portfolio: str) -> pd.DataFrame:

    path = PRICES_DIR / f"{portfolio}_prices.csv"
    prices = pd.read_csv(path, parse_dates=["Date"]).set_index("Date").sort_index()
    prices.columns = prices.columns.astype(str)

    if prices.index.has_duplicates or prices.isna().any().any() or (prices <= 0).any().any():
        raise ValueError(f"Invalid existing portfolio prices: {path}")
    
    if prices.shape[1] != PORTFOLIO_SIZE:
        raise ValueError(f"Expected four assets in {path}")
    
    return prices


def eligible_beverage_ids() -> list[str]:

    metadata_path = SOURCE_DIR / "phase1" / "constituents_metadata_preview.csv"
    coverage_path = SOURCE_DIR / "phase1" / "issue_coverage.csv"

    metadata = pd.read_csv(metadata_path, dtype={"IssueId": str})
    coverage = pd.read_csv(coverage_path, dtype={"IssueId": str})

    eligible = metadata.merge(coverage, on="IssueId", validate="one_to_one")
    eligible = eligible.loc[
        (eligible["Country"] == "US")
        & (eligible["SubIndustryCode"] == SUBINDUSTRY)
        & (eligible["first_date"] == "2014-12-31")
        & (eligible["last_date"] == "2020-12-31")
        & (eligible["valid_prices"] >= 1512)
        & (eligible["missing_prices"] == 0)
    ]

    ids = sorted(set(eligible["IssueId"]) & set(BRANDS))
    if not ANCHORS.issubset(ids):
        raise ValueError(f"Missing mandatory Coca-Cola/PepsiCo IDs: {sorted(ANCHORS - set(ids))}")
    
    if len(ids) < PORTFOLIO_SIZE:
        raise ValueError(f"Only {len(ids)} fully covered beverage securities")
    
    print(f"Metadata/coverage-eligible verified beverage companies: {len(ids)}")
    for iid in ids:
        print(f"  {iid}  {BRANDS[iid][1]} ({BRANDS[iid][2]})")

    return ids


def load_candidate_prices(ids: list[str], reference_dates: pd.DatetimeIndex):

    data = load_sebx_am_data(amgm_config.am_dataset_dir)
    security = data["security_data"]
    members = data["constituents"]

    security = security.loc[security["IssueId"].astype(str).isin(ids), ["IssueId", "Date", "ClAdjLoc"]].copy()
    security["IssueId"] = security["IssueId"].astype(str)
    security["Date"] = pd.to_datetime(security["Date"])

    if security.duplicated(["IssueId", "Date"]).any():
        raise ValueError("Duplicate historical security prices")

    matrix = security.pivot(index="Date", columns="IssueId", values="ClAdjLoc")
    matrix = matrix.reindex(reference_dates)
    matrix = matrix.reindex(columns=ids)
    matrix = matrix.apply(pd.to_numeric, errors="coerce")

    valid_columns = matrix.notna().all() & (matrix > 0).all()
    matrix = matrix.loc[:, valid_columns]
    if matrix.shape[1] < PORTFOLIO_SIZE:
        raise ValueError("Fewer than four securities have complete aligned positive prices")

    members = members.loc[:, ["IssueId", "Gvkey"]].copy()
    members["IssueId"] = members["IssueId"].astype(str)
    if members.duplicated("IssueId").any():
        raise ValueError("Duplicate IssueIds in constituent metadata")
    
    company = members.set_index("IssueId")["Gvkey"].reindex(matrix.columns)
    if company.isna().any():
        missing = company[company.isna()].index.tolist()
        raise ValueError(f"Missing Gvkey for candidate securities: {missing}")

    # Verify external company mapping against the actual dataset's GVKEY.
    for iid in matrix.columns:
        expected_gvkey = BRANDS[iid][0]
        if int(company.loc[iid]) != expected_gvkey:
            raise ValueError(f"Company identity check failed for {iid}: found GVKEY {company.loc[iid]}, expected {expected_gvkey}")
        
    if not ANCHORS.issubset(set(matrix.columns)):
        raise ValueError("Coca-Cola or PepsiCo lacks aligned price history")
    
    print(f"Fully aligned beverage candidates: {matrix.shape[1]}")
    print(f"Reference dates: {len(matrix):,} ({matrix.index.min().date()} to {matrix.index.max().date()})")

    return matrix, company


def choose_quartet(prices: pd.DataFrame, company: pd.Series):

    returns = prices.pct_change(fill_method=None)
    calibration = returns.loc[CALIBRATION_START:CALIBRATION_END]
    if len(calibration) < 400 or calibration.isna().any().any():
        raise ValueError("Insufficient clean calibration returns")

    corr = calibration.corr().to_numpy()
    ids = prices.columns.tolist()
    firm_ids = company.astype(str).tolist()
    scored = []
    checked = 0

    # Evaluate all valid quartets containing Coca-Cola and PepsiCo.
    # Maximize the weakest correlation first, then the average.
    for quartet in combinations(range(len(ids)), PORTFOLIO_SIZE):
        choice = tuple(ids[i] for i in quartet)
        if not ANCHORS.issubset(choice):
            continue
        if len({firm_ids[i] for i in quartet}) < PORTFOLIO_SIZE:
            continue
        pair_corr = [corr[a, b] for a, b in combinations(quartet, 2)]
        if not np.isfinite(pair_corr).all():
            continue
        weakest = float(min(pair_corr))
        mean = float(np.mean(pair_corr))
        checked += 1
        scored.append(((weakest, mean), choice))

    if not scored:
        raise ValueError("No valid four-company portfolio found")

    ranking = sorted(scored, reverse=True)
    best = ranking[0]
    chosen = list(best[1])
    ranking_df = pd.DataFrame(
        [{"Rank": i, "IssueIds": ";".join(quartet),
          "Companies": ";".join(BRANDS[iid][1] for iid in quartet),
          "MinPairCorr_2016_17": score[0], "MeanPairCorr_2016_17": score[1]}
         for i, (score, quartet) in enumerate(ranking, start=1)]
    )
    print(f"Evaluated distinct-company quartets: {checked:,}")
    print("SELECTED PORTFOLIO:")
    for iid in chosen:
        print(f"  {iid}: {BRANDS[iid][1]} ({BRANDS[iid][2]})")
    print(f"2016-17 weakest pairwise correlation: {best[0][0]:.3f}")
    print(f"2016-17 average pairwise correlation: {best[0][1]:.3f}")
    if best[0][0] < 0.70:
        print("WARNING: weakest correlation below 0.70; the selected beverage portfolio is not strongly correlated across all pairs.")

    return prices[chosen].copy(), ranking_df


def make_diagnostics(portfolios: dict[str, pd.DataFrame]):
    records = []
    for portfolio, prices in portfolios.items():
        ret = prices.pct_change(fill_method=None)
        for period, (start, end) in PERIODS.items():
            sample = ret.loc[start:end].dropna(how="any")
            if len(sample) < 30:
                raise ValueError(f"Too few returns for {portfolio} / {period}")
            values = sample.corr().to_numpy()[np.triu_indices(PORTFOLIO_SIZE, k=1)]
            records.append({
                "Portfolio": portfolio,
                "Period": period,
                "N_ReturnDays": len(sample),
                "MeanPairCorrelation": float(values.mean()),
                "MinPairCorrelation": float(values.min()),
                "MaxPairCorrelation": float(values.max())
            })
    return pd.DataFrame(records)


def main():

    diversified = read_existing_prices("diversified")
    concentrated = read_existing_prices("concentrated")
    if not diversified.index.equals(concentrated.index):
        raise ValueError("Existing portfolio calendars differ")

    original_definitions = pd.read_csv(PRICES_DIR / "portfolio_definitions.csv", dtype={"IssueId": str})
    ids = eligible_beverage_ids()
    candidates, companies = load_candidate_prices(ids, diversified.index)
    beverage, ranking = choose_quartet(candidates, companies)

    if not beverage.index.equals(diversified.index):
        raise ValueError("Selected portfolio calendar does not align")

    selected_metadata = pd.read_csv(
        SOURCE_DIR / "phase1" / "constituents_metadata_preview.csv",
        dtype={"IssueId": str},
    ).set_index("IssueId")
    new_rows = []
    for iid in beverage.columns:
        row = selected_metadata.loc[iid]
        new_rows.append({
            "portfolio": "beverage_concentrated", "IssueId": iid,
            "Company": BRANDS[iid][1], "Ticker": BRANDS[iid][2],
            "Country": row["Country"], "SectorCode": int(row["SectorCode"]),
            "Sector": "Consumer Staples", "IndustryCode": int(row["IndustryCode"]),
            "SubIndustryCode": int(row["SubIndustryCode"]),
            "first_date": str(beverage.index.min().date()),
            "last_date": str(beverage.index.max().date()),
            "valid_prices": len(beverage), "missing_prices": 0, "duplicate_dates": 0
        })
    combined = pd.concat([original_definitions, pd.DataFrame(new_rows)], ignore_index=True)
    if combined.groupby("portfolio")["IssueId"].nunique().to_dict() != {
        "diversified": 4, "concentrated": 4, "beverage_concentrated": 4
    }:
        raise ValueError("Portfolio definition counts are not 4/4/4")

    diagnostics = make_diagnostics({
        "diversified": diversified,
        "concentrated": concentrated,
        "beverage_concentrated": beverage
    })
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    beverage.rename_axis("Date").to_csv(OUTPUT_DIR / "beverage_concentrated_prices.csv")
    combined.to_csv(OUTPUT_DIR / "portfolio_definitions_3groups.csv", index=False)
    ranking.to_csv(OUTPUT_DIR / "beverage_candidate_ranking.csv", index=False)
    diagnostics.to_csv(OUTPUT_DIR / "portfolio_period_correlation_preview.csv", index=False)

    print("\nBEVERAGE 2016-2017 RETURN CORRELATIONS")
    print(beverage.pct_change(fill_method=None).loc[CALIBRATION_START:CALIBRATION_END].corr().round(3).to_string())
    print("\nPORTFOLIO / PERIOD CORRELATION PREVIEW")
    print(diagnostics.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print("\nSaved 4 files under:", OUTPUT_DIR)


if __name__ == "__main__":
    main()