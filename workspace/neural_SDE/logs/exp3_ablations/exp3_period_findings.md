# Exp 3 period comparison

Delta best val NLL vs independent (mean +/- std, 6 seeds).
Negative means the joint model fits the held-out days better than independent assets.

| Basket | Period | mean corr | ceiling | Correlation only | Full joint |
|---|---|---|---|---|---|
| concentrated | 2015-2017 | 0.30 | 0.25 | -0.171 +/- 0.159 | -0.353 +/- 0.245 |
| diversified | 2015-2017 | 0.20 | 0.11 | +0.008 +/- 0.186 | -0.141 +/- 0.128 |
| high-correlation | 2015-2017 | 0.70 | 1.28 | -1.412 +/- 0.112 | -1.414 +/- 0.110 |
| concentrated | 2018-2020 | 0.54 | 0.71 | -0.405 +/- 0.244 | -0.402 +/- 0.252 |
| diversified | 2018-2020 | 0.43 | 0.54 | -0.335 +/- 0.093 | -0.358 +/- 0.088 |
| high-correlation | 2018-2020 | 0.78 | 1.75 | -1.532 +/- 0.083 | -1.626 +/- 0.070 |
| concentrated | 2014-2020 | 0.41 | 0.43 | -0.365 +/- 0.139 | -0.388 +/- 0.136 |
| diversified | 2014-2020 | 0.33 | 0.30 | -0.232 +/- 0.115 | -0.237 +/- 0.118 |
| high-correlation | 2014-2020 | 0.74 | 1.47 | -1.306 +/- 0.193 | -1.281 +/- 0.167 |

Figure: `exp3_period_scaling.png`. Data: `exp3_period_comparison.csv`.
