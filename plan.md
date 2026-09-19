Here is the complete blueprint of your project, combining the baseline architecture, the targeted improvements from the whitepaper, and your team's division of labor.

**The Baseline Architecture**

* **Objective:** Generate synthetic financial time series that faithfully reproduce real-world market dynamics and statistical properties.


* **The Engine:** A path-dependent Neural Stochastic Differential Equation (Neural SDE) modeled by $dX_{t}=\mu_{\theta}(X_{[t-w,t]},D_{t})dt+\sigma_{\phi}(X_{[t-w,t]},D_{t})dW_{t}$.


* **Technical Geometry:** The model is explicitly conditioned on recent price history and the normalized distance ($D_{t}$) to dynamically computed Fibonacci retracement levels.


* **Mixture-of-Experts (MoE):** The drift ($\mu$) and diffusion ($\sigma$) networks use a gating mechanism to route decisions through three structural experts modeling localized price behaviors: **Bounce** (mean-reversion), **Break** (directional continuation), and **Hover** (price consolidation).



**The Three Strategic Upgrades**

* **Preventing MoE Collapse:** The current gating network often collapses, routing all training to a single expert instead of utilizing all three. Fixing this ensures the output probabilities remain interpretable for finance professionals.


* **Stabilizing Long-Horizon Rollouts:** The current model generates extreme/unrealistic prices when synthesizing long trajectories because it drifts into out-of-distribution states. The fix requires augmenting the state representation to condition on multiple historical windows (e.g., 252, 756, and 2520 days) simultaneously.


* **Multivariate Joint Generation:** The current model generates paths for single assets in isolation. Upgrading to a multi-asset joint SDE allows the model to capture cross-asset dependencies, making it viable for portfolio risk management tasks like Value-at-Risk (VaR) estimation.



**Team Execution Plan**

* **Member 1 (MoE Optimization):** Focuses exclusively on the gating network and loss functions. They will utilize categorical entropy regularization and balance losses to prevent expert collapse and ensure the Bounce, Break, and Hover probabilities behave predictably.


* **Member 2 (Multi-Scale Path Engineering):** Modifies the data pipeline and sequence encoder. They will update the latent state embedding to process the multiple time horizons (252-day, 756-day, 2520-day) required to stabilize long-term rollouts.


* **Member 3 (Core SDE Architecture - You):** Upgrades the internal mathematics of the Neural SDE. You will refactor the drift ($\mu$) and diffusion ($\sigma$) networks to ingest $N$-dimensional inputs and output multivariate tensors, effectively scaling the model to generate entire correlated portfolios simultaneously.

* **Member 4 (Multivariate Evaluation):** Replaces the standard 1D histograms with entirely new evaluation metrics. They will design statistical tests to prove your $N$-dimensional synthetic data accurately preserves real-world stock correlations.

> **Update (post meeting 2):** Alireza has structured the Member 3+4 work as six experiments
> (dependency analysis → features → joint architecture → marginal realism → cross-asset
> evaluation → portfolio evaluation). The detailed working plan, including the mapping to
> the already-implemented multi-asset pipeline, is in [plan-member3-4-experiments.md](plan-member3-4-experiments.md).



You are member 3:

Your task is to transform the existing single-asset Neural SDE into a multidimensional architecture capable of generating synthetic paths for an entire portfolio simultaneously.

**Core Architecture Upgrades**

* **N-Dimensional Inputs:** Modify the sequence encoder to ingest matrices containing the historical price windows and Fibonacci distances ($D_t$) for $N$ correlated assets rather than an isolated stock.


* **Multivariate Drift and Diffusion:** Expand the expert neural networks so drift ($\mu$) outputs an $N$-dimensional vector (individual asset trends) and diffusion ($\sigma$) outputs an $N \times d$ matrix to capture both individual volatilities and cross-asset correlations.


* **MoE Tensor Routing:** Adapt the gating network and the three structural experts (Bounce, Break, Hover) to evaluate multi-asset boundary interactions and output coordinated regime probabilities.


* **Loss Function Adaptation:** Update the negative log-likelihood calculation in the SDE loss to evaluate a multivariate Gaussian distribution instead of a univariate one.


First draft of the plan (IMPORTANT: you must refine this plan, this is just a draft and not decided yet, but just a starting point, it was made before we got access to the codebase):


**Phase 1: Tensor & Encoder Scaling (Weeks 1-2)**

* Refactor data loaders to supply `[batch, time, N_assets]` tensors rather than isolated 1D price vectors.
* Modify the sequence encoder to ingest and compress $N$-dimensional historical paths and Fibonacci distances ($D_t$) without losing cross-asset context.

**Phase 2: Drift & Diffusion Architecture (Weeks 3-4)**

* Expand the drift ($\mu$) structural experts to output an $N$-dimensional vector representing simultaneous trends for all assets.
* Redesign the diffusion ($\sigma$) experts to output an $N \times d$ lower-triangular matrix (using Cholesky decomposition) to ensure a valid, positive semi-definite covariance matrix for modeling asset correlations.

**Phase 3: MoE Gating & Loss Translation (Weeks 5-6)**

* Implement the routing logic: decide whether the gating network calculates one global regime probability for the whole portfolio, or $N$ independent gating distributions.
* Rewrite the SDE Negative Log-Likelihood (NLL) loss function to mathematically evaluate a multivariate Gaussian distribution.


**Phase 4: Debugging & Integration (Weeks 7-8)**

* Resolve inevitable tensor shape mismatches during backpropagation.
* Pass your generated $N$-dimensional paths to Member 4 to verify that the model actually preserves real-world asset correlations.