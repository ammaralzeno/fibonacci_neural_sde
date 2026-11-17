# Neural SDE Workflow

The detailed Neural SDE workflow is documented in the repository root README.

From the project root, use:

```bash
python experiments/neural_SDE/train_neural_SDE.py --trainer_cfg US_Stocks
python experiments/neural_SDE/generate_samples.py --trainer_cfg US_Stocks --mode mc --batch-size 10000
python experiments/neural_SDE/generate_samples.py --trainer_cfg US_Stocks --batch-size 1000
python experiments/neural_SDE/train_classification.py --trainer_cfg US_Stocks
```

See the top-level `README.md` for setup, generated artifacts, available CLI options, and the end-to-end workflow description.


## Gate inputs (comparing gate architectures)

`model_cfg.gate_inputs` in `trainer_cfg/neural_SDE/US_Stocks.py` selects what the MoE gate sees. Experts always see `h = [Z_t || f_t]`.

| `gate_inputs` | Gate input |
|---|---|
| `"h"` (default) | `h`: the original architecture. Existing checkpoints load unchanged. |
| `"features"` | only the gate features `g_t` from `model_cfg.gate_feature_set` |
| `"h+features"` | `[h || g_t]` |

The feature sets live in `amgm/models/neural_SDE/gate_features.py` (`GATE_FEATURE_SETS`). They are computed inside the model from the normalised window, so training, generation and the gating diagnostics all see identical inputs.

- `price_v1` has 14 features: distances to the 7 levels in volatility units, an approach indicator, 1/5/10-day velocity, `log(σ20/σ60)`, log range/vol, and days in the ±0.02 zone.
- `vol_only` is the volatility-only ablation.

To add a feature set, write a function `(x_hist, **options) -> (B, G)` and register it with its column names and minimum lookback. Then select it with `model_cfg.gate_feature_set`. `model_cfg.gate_feature_options` passes keyword arguments such as `clip` or `zone_tol`.

Benchmark variants use the same dotted overrides as the regularisers, for example `{"model_cfg.gate_inputs": "features"}`. See the "Gate-input variants" cell in `gate_collapse_benchmark.ipynb`.

Exp 2 (`analyze_gating`) writes the `price_v1` features as `gate_*` columns in `real_gates.csv` for every checkpoint, so you can probe whether they predict Fibonacci events.
