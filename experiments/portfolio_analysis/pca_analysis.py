from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = REPO_ROOT / "workspace" / "portfolio_analysis" / "data"
OUTPUT_DIR = REPO_ROOT / "workspace" / "portfolio_analysis" / "results"

DIVERSIFIED_RETURN_FILE = DATA_DIR / "diversified_returns.csv"
CONCENTRATED_RETURN_FILE = DATA_DIR / "concentrated_returns.csv"

def load_return_matrix(file_path):

    if not file_path.exists():
        raise FileNotFoundError(f"Return file not found:\n{file_path}")

    returns = pd.read_csv(file_path, parse_dates=["Date"], index_col="Date")
    returns = returns.sort_index()

    for column in returns.columns:
        returns[column] = pd.to_numeric(returns[column], errors="coerce")

    return returns


def validate_returns(portfolio_name, returns):

    missing = returns.isna().sum().sum()
    non_finite = (~np.isfinite(returns.to_numpy())).sum()

    if missing > 0:
        raise ValueError(f"{portfolio_name}: missing return values found")

    if non_finite > 0:
        raise ValueError(f"{portfolio_name}: non-finite return values found")

    if returns.shape[1] < 2:
        raise ValueError(f"{portfolio_name}: PCA requires multiple securities")


def standardize_returns(returns):

    means = returns.mean()
    stds = returns.std(ddof=1)

    if (stds == 0).any():
        raise ValueError("At least one security has zero return variance")

    standardized = (returns - means) / stds
    return standardized


def calculate_pca(standardized_returns):
    # Performs PCA via Eigen decomp

    correlation_matrix = standardized_returns.cov()

    eigenvalues, eigenvectors = np.linalg.eigh(correlation_matrix.to_numpy())
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    for component_index in range(eigenvectors.shape[1]):
        vector = eigenvectors[:, component_index]
        largest_position = np.argmax(np.abs(vector))

        if vector[largest_position] < 0:
            eigenvectors[:, component_index] *= -1

    explained_variance_ratio = eigenvalues / eigenvalues.sum()
    cumulative_explained_variance = np.cumsum(explained_variance_ratio)

    component_names = [f"PC{i + 1}" for i in range(len(eigenvalues))]
    summary = pd.DataFrame(
        {
            "component": component_names,
            "eigenvalue": eigenvalues,
            "explained_variance_ratio": explained_variance_ratio,
            "cumulative_explained_variance_ratio": cumulative_explained_variance
        }
    )

    component_weights = pd.DataFrame(eigenvectors, index=standardized_returns.columns, columns=component_names)
    component_weights.index.name = "IssueId"

    loadings = component_weights.multiply(np.sqrt(eigenvalues), axis=1)

    return summary, component_weights, loadings
    

def print_pca_results(portfolio_name, summary, component_weights, loadings):

    print("\n" + "=" * 80)
    print(f"PCA RESULTS: {portfolio_name.upper()}")
    print("=" * 80)

    display_summary = summary.copy()
    display_summary["explained_variance_pct"] = display_summary["explained_variance_ratio"] * 100
    display_summary["cumulative_variance_pct"] = display_summary["cumulative_explained_variance_ratio"] * 100

    print(display_summary[
        [
            "component",
            "eigenvalue",
            "explained_variance_pct",
            "cumulative_variance_pct"
        ]
    ].to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print("\nComponent weights:")
    print(component_weights.to_string(float_format=lambda x: f"{x:.4f}"))

    print("\nPCA loadings:")
    print(loadings.to_string(float_format=lambda x: f"{x:.4f}"))

    print(f"\nPC1 explained variance: {summary.loc[0, 'explained_variance_ratio'] * 100:.2f}%")
    

def save_results(portfolio_name, summary, component_weights, loadings):

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summary_file = OUTPUT_DIR / f"{portfolio_name}_pca_summary.csv"
    weights_file = OUTPUT_DIR / f"{portfolio_name}_pca_component_weights.csv"
    loadings_file = OUTPUT_DIR / f"{portfolio_name}_pca_loadings.csv"

    summary.to_csv(summary_file, index=False)
    component_weights.to_csv(weights_file, index=True)
    loadings.to_csv(loadings_file, index=True)

    print(f"\nSaved PCA summary:  {summary_file}")
    print(f"Saved component weights:  {weights_file}")
    print(f"Saved PCA loadings:  {loadings_file}")


def process_portfolio(portfolio_name, return_file):

    returns = load_return_matrix(return_file)
    validate_returns(portfolio_name, returns)

    standardized_returns = standardize_returns(returns)
    summary, component_weights, loadings = calculate_pca(standardized_returns)

    print_pca_results(portfolio_name, summary, component_weights, loadings)
    save_results(portfolio_name, summary, component_weights, loadings)


def main():

    print("=" * 80)
    print("PCA / COMMON FACTOR ANALYSIS")
    print("=" * 80)

    process_portfolio(portfolio_name="diversified", return_file=DIVERSIFIED_RETURN_FILE)
    process_portfolio(portfolio_name="concentrated", return_file=CONCENTRATED_RETURN_FILE)

    print("\n" + "=" * 80)
    print("PCA ANALYSIS COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()