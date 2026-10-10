# Experiment 3 — basket comparison: joint-modeling value vs dependency strength

Same ablation protocol on three baskets of increasing cross-asset dependency,
seeds (42, 43, 44, 45, 46, 47) (mean +/- std). Validation NLL: lower is better; delta is vs the
basket's own independent baseline of the same seed.

The correlation ceiling -0.5*logdet(R) upper-bounds the NLL gain learnable from
the basket's correlation structure — it ranks the baskets before any training.

| Basket | mean corr | ceiling | Configuration | best val NLL | delta vs independent |
|---|---|---|---|---|---|
| same-sector | 0.79 | 2.65 | Independent (no context, R=I) | -9.400 +/- 0.477 | +0.000 +/- 0.000 |
| same-sector | 0.79 | 2.65 | Correlation only | -11.769 +/- 0.523 | -2.370 +/- 0.401 |
| same-sector | 0.79 | 2.65 | Context only (R=I) | -9.418 +/- 0.506 | -0.018 +/- 0.039 |
| same-sector | 0.79 | 2.65 | Full joint (context + R) | -11.707 +/- 0.628 | -2.307 +/- 0.466 |
| concentrated | 0.30 | 0.25 | Independent (no context, R=I) | -6.055 +/- 0.953 | +0.000 +/- 0.000 |
| concentrated | 0.30 | 0.25 | Correlation only | -6.226 +/- 1.036 | -0.171 +/- 0.159 |
| concentrated | 0.30 | 0.25 | Context only (R=I) | -6.226 +/- 0.735 | -0.170 +/- 0.265 |
| concentrated | 0.30 | 0.25 | Full joint (context + R) | -6.408 +/- 0.787 | -0.353 +/- 0.245 |
| diversified | 0.20 | 0.11 | Independent (no context, R=I) | -5.999 +/- 0.646 | +0.000 +/- 0.000 |
| diversified | 0.20 | 0.11 | Correlation only | -5.991 +/- 0.795 | +0.008 +/- 0.186 |
| diversified | 0.20 | 0.11 | Context only (R=I) | -6.138 +/- 0.579 | -0.139 +/- 0.098 |
| diversified | 0.20 | 0.11 | Full joint (context + R) | -6.141 +/- 0.659 | -0.141 +/- 0.128 |

Figure: `exp3_basket_scaling.png`. Data: `exp3_basket_comparison.csv`.

Caveats: ~400 train / ~100 val windows per basket; deltas within one seed-std
of zero should be read as no measurable effect.
