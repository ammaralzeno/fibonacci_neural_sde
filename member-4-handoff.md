# Handoff to Member 4 — Multi-Asset Synthetic Data (from Member 3)

## What you will receive

1. **Trained multi-asset MoE Neural SDE checkpoint**
   `workspace/neural_SDE/logs/train_neural_SDE_multi/US_Stocks_Multi/checkpoints/*.ckpt`
2. **Correlated Monte Carlo rollouts** for a fixed basket of N US stocks (default N = 10, configurable), produced by `experiments/neural_SDE/generate_samples_multi.py`.
3. **A correlation sanity plot** (real vs synthetic daily-return correlation heatmaps + mean absolute error of off-diagonal entries). This is a quick proof that the joint model works — the full multivariate evaluation suite is yours.

## Artifact 1: NPZ — `Data/Synthetic/synthetic_rollout_MC_multi_valid_samples.npz`

| Key | Dtype | Shape | Meaning |
|---|---|---|---|
| `synthetic_paths` | float32 | `(n_conditions * mc_paths, n_steps+1, N)` | Correlated synthetic price paths, original price scale |
| `basket_issue_ids` | str | `(N,)` | Asset order matching axis 2 of `synthetic_paths` |
| `seed_windows` | float32 | `(n_conditions, 252, N)` | Real seed windows the rollouts start from (original scale) |
| `original_prices` | float32 | `(n_conditions, n_steps+1, N)` | Real future paths for the same start dates (NaN-padded) |
| `fib_levels` | float32 | `(n_conditions, N, 7)` | Fibonacci levels per asset at each forecast origin |
| `test_dates` | str | `(n_conditions,)` | Forecast origin per initial condition |
| `is_valid` | bool | `(n_conditions * mc_paths,)` | Per-path sanity-check result |
| `corr_matrix` | float32 | `(N, N)` | The model's learned correlation matrix R |
| `master_seed` | int64 | `(1,)` | RNG seed for reproducibility |

## Artifact 2: CSV — `Data/Synthetic/synthetic_rollout_paths_MultiMoE.csv` (generated on demand, not committed)

Long format, one row per asset per day per path. Mirrors the existing single-asset CSVs (`synthetic_rollout_paths_MoE.csv` etc.), so existing loaders keep working:

`IssueId` (synthetic, includes path/iteration), `SourceIssueId` (real asset), `AssetIdx` (0..N-1), `TestDate`, `PathId`, `MCIteration`, `MasterSeed`, `IsValid`, `Date`, `ClAdjLoc`.

**Not in git** (17.8 MB flattened copy of the NPZ — the NPZ above is the canonical committed artifact). Regenerate it (and the NPZ) with:

```bash
python experiments/neural_SDE/generate_samples_multi.py --trainer_cfg US_Stocks_Multi --seed 42
```

## What stays on your side

- Multivariate evaluation: correlation-preservation metrics, JS distances, portfolio-level statistics, VaR backtest.
- Extending `fibonacci_state_detection.py` and `Statistical_comparison.ipynb` to N assets — I do not touch those files.
- The existing single-asset artifacts in `Data/Synthetic/` remain valid baselines and show the established artifact pattern.

## Notes

- The basket is fixed per run and recorded in `basket_issue_ids`. Tell me before I retrain if you need a different basket or N.
- Correlation is modeled as a learned constant matrix R with per-asset regime-dependent volatilities (Σ_t = diag(σ_t)·R·diag(σ_t)).
