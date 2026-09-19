# Member 3 & 4 — Experiment Plan (Alireza's structure, post meeting 2)

Alireza has structured the remaining work for Members 3 and 4 as six experiments.
This document is the working plan: what each experiment needs, what already exists
from Member 3's implementation (Phases 1–5, PRs #2–#6), and what is new.

## Goal (as given)

Extend the single-asset formulation to a joint multi-asset system
`X_t = [X^(1)_t, ..., X^(N)_t]` that learns both:

- **within-asset dynamics** — e.g. Fibonacci behavior;
- **cross-asset dependencies** — e.g. correlation, lead/lag relationships, common movements.

This is exactly the architecture already implemented: per-asset Fibonacci MoE
(shared-tower encoder, per-asset gates/experts) + learned correlation matrix R
(`Sigma_t = diag(sigma_t) R diag(sigma_t)`) + shared portfolio context vector.

## Experiment mapping: what exists vs what is new

| # | Experiment | Already built (Member 3, merged/PR'd) | New work |
|---|---|---|---|
| 1 | Cross-asset dependency analysis | Date-aligned basket pipeline (`MultiAssetNeuralSDEDataset.prices_aligned`), increment-correlation init | Analysis script: correlation matrix, rolling correlations, lagged cross-correlation, co-movement during large moves, PCA/common factors |
| 2 | Multi-asset Fibonacci features | `_fib_features_multi` — per-asset levels + normalized signed distances (tested) | Simultaneous multi-asset visualization; simultaneity/coincidence analysis (do assets reach levels together; do fib events in A coincide with moves in B) |
| 3 | Joint architecture | Full joint model + multivariate Gaussian NLL runner + training pipeline + configs (`US_Stocks_Multi`, `synthetic_gbm_multi`) | Ablation flags in `NeuralSDEMoEMultiAsset` (`use_context`, fixed R=I vs learned R); train 4 configurations (independent / corr-only / context-only / full); validation NLL + convergence table |
| 4 | Marginal realism | Correlated MC generation pipeline (`generate_samples_multi.py`); mean return / volatility / MDD already computed ad hoc (real vs synthetic) | Full per-asset suite: + autocorrelation, return distribution, Bounce/Break/Hover distributions (reuse `fibonacci_state_detection.py` per asset), fib-distance distributions; Real vs Independent vs Joint harness |
| 5 | Cross-asset dependency evaluation | Joint generation machinery; committed NPZ/CSV artifacts | Conditional analysis: given `|D^(A)_t| < eps`, distribution of B's returns; cross-correlation and tail dependence real vs synthetic; possible architecture enhancement (see risk below) |
| 6 | Portfolio-level evaluation | The correlated trajectories themselves (NPZ schema committed) | Portfolio aggregation (equal-weight baseline): volatility, drawdown, Sharpe, return distribution, VaR, Expected Shortfall, real vs synthetic; downstream tasks (regime classification, VaR estimation, portfolio forecasting) |

## Design risk flagged for Experiment 5

Alireza explicitly wants to test whether the joint model captures interaction
between Fibonacci regimes across assets, "rather than merely reproducing a static
correlation matrix". Our current cross-asset channels are:

1. a learned **constant** correlation matrix R in the diffusion (static by design);
2. a mean-pooled **portfolio context** vector in every per-asset gate/expert
   (state-dependent, but indirect).

If the Exp 5 conditional analysis shows channel 2 is too weak, the planned
enhancement (small, contained): add explicit cross-asset Fibonacci features to the
per-asset gate/expert input — e.g. the basket's min `|D|` or a reference/market
asset's signed distance — or upgrade R to a regime-dependent R_t. Decision deferred
to after Exp 3/5 results and the next meeting (see questions).

## Division of labor (proposal)

- **Member 3**: Experiments 1, 2, 3 (data analysis, features, architecture + ablations),
  plus any architecture enhancement required by Experiment 5.
- **Member 4**: Experiments 4, 5, 6 (evaluation suites), building on the committed
  artifacts and generator; extends `fibonacci_state_detection.py` per asset.
- Shared: Experiment 5 design review before implementation.

## Execution order (dependency-driven)

- **Phase A** (M3): Exp 1 + Exp 2 analysis scripts — pure data work on the aligned basket.
- **Phase B** (M3): Exp 3 ablation flags + 4-config training + NLL/convergence table.
- **Phase C** (M4): Exp 4 marginal realism suite (needs Phase 5 artifacts; independent
  baseline from Phase B).
- **Phase D** (M4 + M3 support): Exp 5 cross-asset evaluation; M3 implements the
  enhancement if the context pathway proves too weak.
- **Phase E** (M4): Exp 6 portfolio evaluation + downstream tasks.

## Open questions for Alireza (next meeting)

1. Is the Member 3/4 split above (3: Exp 1–3 + architecture; 4: Exp 4–6) as intended?
2. Exp 3 baseline: should "Independent NSDE" be his original single-asset model, or our
   multi-asset model with R=I and no context? Recommendation: ours (isolates the joint
   mechanism from architecture differences), additionally reporting his single-asset
   numbers for continuity with the paper.
3. Exp 5: is a static learned R + context-conditioned drift/volatility acceptable as
   "interaction", or does he expect regime-dependent correlation R_t? His wording
   suggests the latter — this decides whether the enhancement is in scope.
4. Basket size: his Exp 1 suggests 3–5 assets; our current default is 10. Recommendation:
   run Exp 1/2 analysis on 5 assets for interpretability, keep N configurable for
   generation. Confirm.
5. Exp 6 downstream task: extend the paper's barrier-touch classifier per asset, or
   portfolio-level tasks only (VaR estimation, portfolio forecasting)?
