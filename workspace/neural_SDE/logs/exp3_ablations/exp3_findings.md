# Experiment 3 — Joint architecture ablations: findings

Same 5-asset basket, same seed (42), 20 epochs each. Validation NLL is the
multivariate Gaussian NLL (`val/loss_sde`); lower is better.

| Configuration | context | learned R | best val NLL | best epoch | final val NLL | Δ vs independent |
|---|---|---|---|---|---|---|
| Independent (no context, R=I) | False | False | -8.2263 | 18 | -8.2064 | +0.0000 |
| Correlation only | False | True | -8.3093 | 19 | -8.3093 | -0.0829 |
| Context only (R=I) | True | False | -7.9284 | 18 | -7.8396 | +0.2980 |
| Full joint (context + R) | True | True | -7.9828 | 18 | -7.9416 | +0.2435 |

## Mechanism attribution (best val NLL differences, nats)

- Correlation channel: independent -> corr-only = -0.0829
- Context channel: independent -> context-only = +0.2980
- Both (full joint): independent -> full = +0.2435

Figures: `exp3_convergence.png`, `exp3_corr_heatmaps.png`. Data: `exp3_nll_table.csv`.

Caveats: single seed; small sample (380 train / 95 val windows); the basket was
selected by data coverage, and Exp 2 found near-independent cross-asset level
simultaneity on it — mechanism values may grow on a relationship-driven basket
(Member 4's Exp 1).
