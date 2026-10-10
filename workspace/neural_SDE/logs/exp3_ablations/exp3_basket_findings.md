# Exp 3 basket comparison

Delta best val NLL vs independent (mean +/- std, 6 seeds; negative = joint helps).
Ceiling = -0.5*logdet(R): the NLL gain the basket's correlation structure can explain.

| Basket | mean corr | ceiling | Correlation only | Context only | Full joint |
|---|---|---|---|---|---|
| same-sector | 0.79 | 2.65 | -2.37 +/- 0.40 | -0.02 +/- 0.04 | -2.31 +/- 0.47 |
| concentrated | 0.30 | 0.25 | -0.17 +/- 0.16 | -0.17 +/- 0.26 | -0.35 +/- 0.24 |
| diversified | 0.20 | 0.11 | +0.01 +/- 0.19 | -0.14 +/- 0.10 | -0.14 +/- 0.13 |

Gain tracks the ceiling: decisive on same-sector, within noise on Member 4's (2015-2017) baskets.
Figure: `exp3_basket_scaling.png`. Data: `exp3_basket_comparison.csv`.
