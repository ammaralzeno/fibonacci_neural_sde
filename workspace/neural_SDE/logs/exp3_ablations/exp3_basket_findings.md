# Experiment 3 — basket comparison: joint-modeling value vs dependency strength

Same ablation protocol (seed 42, 20 epochs) on three baskets of increasing
cross-asset dependency. Lower validation NLL is better; delta is vs the basket's
own independent baseline.

| Basket | mean pairwise corr | Configuration | best val NLL | delta vs independent |
|---|---|---|---|---|
| same-sector | 0.79 | Independent (no context, R=I) | -9.1514 | +0.0000 |
| same-sector | 0.79 | Correlation only | -10.9087 | -1.7573 |
| same-sector | 0.79 | Context only (R=I) | -9.1645 | -0.0130 |
| same-sector | 0.79 | Full joint (context + R) | -10.6735 | -1.5221 |
| concentrated | 0.30 | Independent (no context, R=I) | -7.1037 | +0.0000 |
| concentrated | 0.30 | Correlation only | -7.1484 | -0.0447 |
| concentrated | 0.30 | Context only (R=I) | -7.1012 | +0.0025 |
| concentrated | 0.30 | Full joint (context + R) | -7.1108 | -0.0071 |
| diversified | 0.20 | Independent (no context, R=I) | -6.5661 | +0.0000 |
| diversified | 0.20 | Correlation only | -6.6958 | -0.1296 |
| diversified | 0.20 | Context only (R=I) | -6.6153 | -0.0491 |
| diversified | 0.20 | Full joint (context + R) | -6.6780 | -0.1119 |

Figure: `exp3_basket_scaling.png`. Data: `exp3_basket_comparison.csv`.

Caveats: single seed; small sample (~400 train / ~100 val windows per basket).
