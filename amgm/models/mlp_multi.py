from torch import nn
import torch
import torch.nn.functional as F

from amgm.models.mlp import ExpertHead, GateHead


class NeuralSDEMoEMultiAsset(nn.Module):
    """
    Multi-asset MoE Neural SDE with the same structural constraints as the
    single-asset baseline, applied per asset:
      - Bounce Expert (Elastic restoring force relative to each asset's closest Fib level)
      - Break Expert (Momentum continuation per asset)
      - Hover Expert (Bounded drift, low volatility per asset)

    Cross-asset dependence enters through (i) a shared portfolio context vector in
    every per-asset gate/expert input and (ii) a learned constant correlation
    matrix R in the diffusion: Sigma_t = diag(sigma_t) R diag(sigma_t).

    Expects:
        x_hist: (batch, lookback_window, n_assets)  per-asset normalized price windows
        f_t:    (batch, n_assets, num_features)     per-asset signed Fibonacci distances
    Returns:
        mu:         (batch, n_assets)
        chol_sigma: (batch, n_assets, n_assets) lower-triangular Cholesky factor of Sigma_t
        pi:         (batch, n_assets, 3)
    """

    def __init__(self, lookback_window, n_assets, num_features=7, hidden_sizes=(64, 32), gate_temperature=1.0, corr_init=None, use_context=True, learn_corr=True):
        super().__init__()
        self.gate_temperature = gate_temperature

        self.lookback_window = lookback_window
        self.n_assets = n_assets
        self.num_features = num_features
        # Ablation flags (Experiment 3): use_context=False removes the portfolio
        # context from every gate/expert input; learn_corr=False freezes R = I.
        self.use_context = use_context
        self.learn_corr = learn_corr

        # Shared per-asset encoder tower (same architecture as the single-asset baseline)
        layers = [nn.Linear(lookback_window, hidden_sizes[0]), nn.ReLU()]
        for i in range(len(hidden_sizes) - 1):
            layers += [nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]), nn.ReLU()]
        self.encoder = nn.Sequential(*layers)

        # Per-asset fused state: [Z_i || D_i || portfolio context]
        h_dim = 2 * hidden_sizes[-1] + num_features

        # 1. Gating Network, shared across assets (outputs 3 logits per asset)
        self.gate_head = GateHead(h_dim, hidden_dim=64, num_experts=3)

        # 2. Expert Heads for Drift Magnitude, shared across assets
        self.mu_bounce_head = ExpertHead(h_dim, hidden_dim=32, out_dim=1)
        self.mu_break_head  = ExpertHead(h_dim, hidden_dim=32, out_dim=1)
        self.mu_hover_head  = ExpertHead(h_dim, hidden_dim=32, out_dim=1)

        # 3. Expert Heads for Diffusion, shared across assets
        self.sigma_bounce_head = ExpertHead(h_dim, hidden_dim=32, out_dim=1)
        self.sigma_break_head  = ExpertHead(h_dim, hidden_dim=32, out_dim=1)
        self.sigma_hover_head  = ExpertHead(h_dim, hidden_dim=32, out_dim=1)

        # 4. Learned constant correlation: R = L_R L_R^T, where L_R is the
        #    row-normalized lower-triangular Cholesky factor. Initialized from the
        #    sample correlation of training increments when provided. With
        #    learn_corr=False the factor is frozen at the identity (R = I).
        if not learn_corr:
            chol = torch.eye(n_assets)
        elif corr_init is not None:
            corr = torch.as_tensor(corr_init, dtype=torch.float32)
            chol = self._safe_cholesky(corr)
        else:
            chol = torch.eye(n_assets)
        self.chol_corr_param = nn.Parameter(chol, requires_grad=learn_corr)

    @staticmethod
    def _safe_cholesky(corr):
        """Cholesky of a correlation matrix, adding diagonal jitter until positive definite."""
        eye = torch.eye(corr.shape[0])
        for jitter in (0.0, 1e-6, 1e-4, 1e-2):
            try:
                return torch.linalg.cholesky(corr + jitter * eye)
            except Exception:
                continue
        return torch.eye(corr.shape[0])

    def _normalized_chol(self):
        """Row-normalized lower-triangular factor: unit-norm rows => valid correlation matrix."""
        L = torch.tril(self.chol_corr_param)
        return L / L.norm(dim=-1, keepdim=True).clamp_min(1e-8)

    def correlation_matrix(self):
        """Current correlation matrix R = L_R L_R^T (unit diagonal, PSD by construction)."""
        L_R = self._normalized_chol()
        return L_R @ L_R.T

    def forward(self, x_hist, f_t):
        batch, _, n_assets = x_hist.shape

        # -------------------------------------------------------------
        # Step A: Identify Closest Fib Level & Direction for Bounce (per asset)
        # -------------------------------------------------------------
        abs_distances = torch.abs(f_t)  # (batch, n_assets, 7)
        closest_idx = torch.argmin(abs_distances, dim=-1, keepdim=True)  # (batch, n_assets, 1)
        closest_signed_dist = torch.gather(f_t, -1, closest_idx).squeeze(-1)  # (batch, n_assets)

        # Direction of bounce per asset:
        # If X_t > L_k (signed_dist > 0), bounce pushes UP (+1)
        # If X_t < L_k (signed_dist < 0), bounce pushes DOWN (-1)
        bounce_sign = torch.sign(closest_signed_dist)
        bounce_sign = torch.where(bounce_sign == 0, torch.ones_like(bounce_sign), bounce_sign)  # Zero handling: treat as +1 (upward bounce)

        # -------------------------------------------------------------
        # Step B: Identify Recent Momentum Direction for Break (per asset)
        # -------------------------------------------------------------
        # Recent velocity from the end of lookback window: X_t - X_{t-1}
        recent_velocity = x_hist[:, -1, :] - x_hist[:, -2, :]  # (batch, n_assets)
        break_sign = torch.sign(recent_velocity)
        break_sign = torch.where(break_sign == 0, torch.ones_like(break_sign), break_sign)  # Zero handling: treat as +1 (upward break)

        # -------------------------------------------------------------
        # Step C: Encoder Feature Extraction & Gating
        # -------------------------------------------------------------
        # 1. Shared temporal encoder processes each asset's history separately
        flat = x_hist.transpose(1, 2).reshape(batch * n_assets, self.lookback_window)  # (batch * n_assets, lookback_window)
        Z = self.encoder(flat).view(batch, n_assets, -1)  # (batch, n_assets, hidden_dim)

        # 2. Portfolio context: mean over assets (permutation-invariant market state)
        context = Z.mean(dim=1, keepdim=True).expand(-1, n_assets, -1)  # (batch, n_assets, hidden_dim)
        if not self.use_context:
            context = torch.zeros_like(context)  # ablation: no cross-asset information

        # 3. Concatenate latent path vector with instantaneous features and context
        h = torch.cat([Z, f_t, context], dim=-1).reshape(batch * n_assets, -1)  # (batch * n_assets, h_dim)

        # 4. Pass concatenated vector 'h' to gating head and expert heads
        pi = F.softmax(self.gate_head(h) / self.gate_temperature, dim=-1).view(batch, n_assets, 3)
        pi_bounce = pi[..., 0]
        pi_break  = pi[..., 1]
        pi_hover  = pi[..., 2]

        # -------------------------------------------------------------
        # Step D: Expert Drift Outputs (per asset)
        # -------------------------------------------------------------
        mu_bounce = bounce_sign * F.softplus(self.mu_bounce_head(h)).view(batch, n_assets)  # Bounce Expert: Direction * positive
        mu_break  = break_sign * F.softplus(self.mu_break_head(h)).view(batch, n_assets)    # Break Expert: Momentum direction * positive
        mu_hover  = 0.05 * torch.tanh(self.mu_hover_head(h)).view(batch, n_assets)          # Hover Expert: Strongly damped drift near zero

        # MoE Drift: \mu_i = \sum_m \pi_{i,m} * \mu_{i,m}
        mu_out = pi_bounce * mu_bounce + pi_break * mu_break + pi_hover * mu_hover

        # -------------------------------------------------------------
        # Step E: Expert Diffusion Outputs (per asset)
        # -------------------------------------------------------------
        sigma_bounce = (F.softplus(self.sigma_bounce_head(h)) + 1e-3).view(batch, n_assets)
        sigma_break  = (F.softplus(self.sigma_break_head(h)) + 1e-3).view(batch, n_assets)
        sigma_hover  = (0.001 + 0.02 * torch.sigmoid(self.sigma_hover_head(h))).view(batch, n_assets)  # Hover diffusion: volatility compression

        # MoE Variance: \sigma_i^2 = \sum_m \pi_{i,m} * \sigma_{i,m}^2
        var_out = pi_bounce * (sigma_bounce**2) + pi_break * (sigma_break**2) + pi_hover * (sigma_hover**2)
        sigma_out = torch.sqrt(var_out)  # (batch, n_assets)

        # -------------------------------------------------------------
        # Step F: Joint covariance Cholesky factor: chol(Sigma_t) = diag(sigma_t) L_R
        # -------------------------------------------------------------
        L_R = self._normalized_chol()  # (n_assets, n_assets)
        chol_sigma = torch.diag_embed(sigma_out) @ L_R  # (batch, n_assets, n_assets)

        return mu_out, chol_sigma, pi
