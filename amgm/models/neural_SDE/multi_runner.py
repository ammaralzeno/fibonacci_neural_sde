import math

import torch

from amgm.models.neural_SDE.runner import NeuralSDERunner


class MultiAssetNeuralSDERunner(NeuralSDERunner):
    """NeuralSDERunner for the multi-asset joint SDE.

    Same training loop and MoE regularizers as the single-asset baseline; only
    the batch shapes and the step NLL change: the one-step transition is a
    multivariate Gaussian evaluated through L_t = chol(Sigma_t) * sqrt(dt).

    Batch: price_window (batch, w, N), nxt_price (batch, N), features (batch, N, 7).
    Model returns mu (batch, N), chol_sigma (batch, N, N), pi (batch, N, 3).
    """

    def _prepare_batch(self, batch):
        x_window = batch.price_window       # x_window = X[t-w, ..., t], (batch, w, N)
        x_t = x_window[:, -1, :]            # x_t = X[t], (batch, N)
        x_tp1 = batch.nxt_price             # x_tp1 = X[t+1], (batch, N)
        f_t = batch.features                # (batch, N, 7)
        fib_levels = batch.fib_levels       # (batch, N, 7)

        return x_window, f_t, x_t, x_tp1, fib_levels

    def _step_nll(self, x_t, x_tp1, mu, chol_sigma):
        """Multivariate Gaussian step NLL: 0.5 * (N log(2pi) + log|Sigma dt| + dx' (Sigma dt)^-1 dx).

        Evaluated via the Cholesky factor L = chol(Sigma) * sqrt(dt): triangular
        solves for the Mahalanobis term, log|Sigma dt| = 2 * sum(log(diag(L))).
        Never inverts or re-factorizes Sigma.
        """
        sqrt_dt = math.sqrt(self.default_dt)
        n_assets = x_t.shape[-1]

        L = chol_sigma * sqrt_dt
        L = L + self.eps * torch.eye(n_assets, device=L.device, dtype=L.dtype)  # diagonal jitter for numerical stability

        dx = (x_tp1 - x_t).unsqueeze(-1)            # (batch, N, 1)
        mean = (mu * self.default_dt).unsqueeze(-1)
        z = torch.linalg.solve_triangular(L, dx - mean, upper=False)  # (batch, N, 1)

        mahalanobis = z.squeeze(-1).pow(2).sum(dim=-1)                       # (batch,)
        log_det = 2.0 * torch.log(L.diagonal(dim1=-2, dim2=-1)).sum(dim=-1)  # (batch,)

        nll = 0.5 * (n_assets * math.log(2.0 * math.pi) + log_det + mahalanobis)
        return nll.mean()

    def _compute_loss(self, batch, entropy_beta=None):
        x_window, f_t, x_t, x_tp1, fib_levels = self._prepare_batch(batch)
        mu, chol_sigma, pi = self.forward(x_window, f_t)
        sde_loss = self._step_nll(x_t, x_tp1, mu, chol_sigma)

        # Same MoE regularizers as the baseline, on gates flattened over assets
        pi_flat = pi.reshape(-1, pi.shape[-1])

        entropy_per_sample = -torch.sum(pi_flat * torch.log(pi_flat + 1e-8), dim=-1)
        mean_entropy = torch.mean(entropy_per_sample)

        f_m = torch.mean(pi_flat, dim=0)
        balance_loss = torch.sum((f_m - (1.0 / 3.0)) ** 2)
        pi_var_per_expert = torch.var(pi_flat, dim=0, unbiased=False)
        mean_pi_var = torch.mean(pi_var_per_expert)
        beta = self.entropy_beta if entropy_beta is None else float(entropy_beta)
        entropy_term = beta * mean_entropy
        balance_term = self.expert_balance_lambda * balance_loss

        loss = sde_loss - entropy_term + balance_term

        x_tp1_pred = x_t + mu * self.default_dt     # mean prediction for x_tp1
        sigma = chol_sigma.diagonal(dim1=-2, dim2=-1)  # per-asset vols, (batch, N)

        return {
            "loss": loss,
            "sde_loss": sde_loss,
            "entropy_term": entropy_term,
            "balance_term": balance_term,
            "mean_entropy": mean_entropy,
            "balance_loss": balance_loss,
            "mean_pi_var": mean_pi_var,
            "pi_mean": f_m,
            "pi_var": pi_var_per_expert,
            "beta": beta,
            "x_window": x_window,
            "features": f_t,
            "x_t": x_t,
            "x_tp1": x_tp1,
            "x_tp1_pred": x_tp1_pred,
            "fib_levels": fib_levels,
            "mu": mu,
            "sigma": sigma,
            "chol_sigma": chol_sigma,
        }

    def on_train_epoch_end(self):
        # Track the learned correlation structure once per epoch
        corr = self.model.correlation_matrix()
        off_diag = corr - torch.diag_embed(corr.diagonal())
        self.log("train/corr_offdiag_abs_mean", off_diag.abs().mean(), prog_bar=False, on_step=False, on_epoch=True)
        self.log("train/corr_offdiag_abs_max", off_diag.abs().max(), prog_bar=False, on_step=False, on_epoch=True)

    def predict_step(self, batch, batch_idx):
        out = self._compute_loss(batch)
        sigma = out["sigma"] + self.eps

        result = {
            "test_dates": batch.test_dates,
            "x_window": out["x_window"].detach().cpu(),
            "features": out["features"].detach().cpu(),
            "x_t": out["x_t"].detach().cpu(),
            "x_tp1": out["x_tp1"].detach().cpu(),
            "x_tp1_pred": out["x_tp1_pred"].detach().cpu(),
            "fib_levels": out["fib_levels"].detach().cpu(),
            "drift": out["mu"].detach().cpu(),
            "diffusion": sigma.detach().cpu(),
            "chol_sigma": out["chol_sigma"].detach().cpu(),
            "batch_loss": out["loss"].detach().cpu(),
        }
        return result
