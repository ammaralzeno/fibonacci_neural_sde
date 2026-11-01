# Exp 3 findings — same-sector basket, seed 42 (detail)

| Configuration | best val NLL | delta vs independent |
|---|---|---|
| Independent (no context, R=I) | -9.151 | +0.000 |
| Correlation only | -10.909 | -1.757 |
| Context only (R=I) | -9.164 | -0.013 |
| Full joint (context + R) | -10.674 | -1.522 |

Correlation channel: -1.757 nats; context channel: -0.013.
Multi-seed, multi-basket version: `exp3_basket_findings.md`.
Figures: `exp3_convergence.png`, `exp3_corr_heatmaps.png`. Data: `exp3_nll_table.csv`.
