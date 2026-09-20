# Experiment 3 — Joint architecture ablations: findings

Same 5-asset basket, same seed (42), 20 epochs each. Validation NLL is the
multivariate Gaussian NLL (`val/loss_sde`); lower is better.

| Configuration | context | learned R | best val NLL | best epoch | final val NLL | Δ vs independent |
|---|---|---|---|---|---|---|
| Independent (no context, R=I) | False | False | -9.1514 | 16 | -9.0622 | +0.0000 |
| Correlation only | False | True | -10.9087 | 4 | -8.9129 | -1.7573 |
| Context only (R=I) | True | False | -9.1645 | 18 | -9.0002 | -0.0130 |
| Full joint (context + R) | True | True | -10.6735 | 4 | -9.3722 | -1.5221 |

## Mechanism attribution (best val NLL differences, nats)

- Correlation channel: independent -> corr-only = -1.7573
- Context channel: independent -> context-only = -0.0130
- Both (full joint): independent -> full = -1.5221

Figures: `exp3_convergence.png`, `exp3_corr_heatmaps.png`. Data: `exp3_nll_table.csv`.

Caveats: single seed; small sample (~400 train / ~100 val windows). The basket is
relationship-selected (same sector, mean pairwise return corr 0.84) and may be
revised after Member 4's Exp 1 dependency analysis.
