# Experiment 3 — period comparison: does the correlation gain follow the regime?

Member 4's portfolios trained on three periods, seeds (42, 43, 44, 45, 46, 47) (mean +/- std).
Rolling correlations are ~50-65% stronger in 2018-2020 than 2015-2017 (Exp 1),
so the correlation ceiling -0.5*logdet(R) rises accordingly.

| Basket | Period | mean corr | ceiling | Correlation only | Full joint |
|---|---|---|---|---|---|
| concentrated | 2015-2017 | 0.30 | 0.25 | -0.171 +/- 0.159 | -0.353 +/- 0.245 |
| diversified | 2015-2017 | 0.20 | 0.11 | +0.008 +/- 0.186 | -0.141 +/- 0.128 |
| concentrated | 2018-2020 | 0.54 | 0.71 | -0.405 +/- 0.244 | -0.402 +/- 0.252 |
| diversified | 2018-2020 | 0.43 | 0.54 | -0.335 +/- 0.093 | -0.358 +/- 0.088 |
| concentrated | 2014-2020 | 0.41 | 0.43 | -0.365 +/- 0.139 | -0.388 +/- 0.136 |
| diversified | 2014-2020 | 0.33 | 0.30 | -0.232 +/- 0.115 | -0.237 +/- 0.118 |

Figure: `exp3_period_scaling.png`. Data: `exp3_period_comparison.csv`.
