# Experiment 3 — basket comparison: joint-modeling value vs dependency strength

Same ablation protocol on three baskets of increasing cross-asset dependency,
seeds (42, 43, 44) (mean +/- std). Validation NLL: lower is better; delta is vs the
basket's own independent baseline of the same seed.

The correlation ceiling -0.5*logdet(R) upper-bounds the NLL gain learnable from
the basket's correlation structure — it ranks the baskets before any training.

| Basket | mean corr | ceiling | Configuration | best val NLL | delta vs independent |
|---|---|---|---|---|---|
| same-sector | 0.79 | 2.65 | Independent (no context, R=I) | -9.054 +/- 0.174 | +0.000 +/- 0.000 |
| same-sector | 0.79 | 2.65 | Correlation only | -11.457 +/- 0.469 | -2.403 +/- 0.468 |
| same-sector | 0.79 | 2.65 | Context only (R=I) | -9.055 +/- 0.191 | -0.001 +/- 0.017 |
| same-sector | 0.79 | 2.65 | Full joint (context + R) | -11.335 +/- 0.569 | -2.281 +/- 0.561 |
| concentrated | 0.30 | 0.25 | Independent (no context, R=I) | -6.642 +/- 0.346 | +0.000 +/- 0.000 |
| concentrated | 0.30 | 0.25 | Correlation only | -6.814 +/- 0.280 | -0.172 +/- 0.097 |
| concentrated | 0.30 | 0.25 | Context only (R=I) | -6.672 +/- 0.326 | -0.030 +/- 0.024 |
| concentrated | 0.30 | 0.25 | Full joint (context + R) | -6.824 +/- 0.247 | -0.183 +/- 0.127 |
| diversified | 0.20 | 0.11 | Independent (no context, R=I) | -6.552 +/- 0.091 | +0.000 +/- 0.000 |
| diversified | 0.20 | 0.11 | Correlation only | -6.647 +/- 0.131 | -0.095 +/- 0.043 |
| diversified | 0.20 | 0.11 | Context only (R=I) | -6.632 +/- 0.110 | -0.080 +/- 0.029 |
| diversified | 0.20 | 0.11 | Full joint (context + R) | -6.695 +/- 0.146 | -0.143 +/- 0.059 |

Figure: `exp3_basket_scaling.png`. Data: `exp3_basket_comparison.csv`.

Caveats: ~400 train / ~100 val windows per basket; deltas within one seed-std
of zero should be read as no measurable effect.
