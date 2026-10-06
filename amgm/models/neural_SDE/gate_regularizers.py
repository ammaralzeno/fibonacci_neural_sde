"""Gate regularisers for the MoE neural SDE.

Every regulariser maps gate probabilities ``pi`` of shape ``(B, K)`` and a scalar
coefficient ``beta`` to a *reward* term ``R``. The runner minimises

    loss = L_SDE - R(pi; beta) + lambda * L_bal

Notation: ``pbar_k = mean_b pi_k(x_b)`` (batch utilisation), ``u = 1/K``,
``H(p) = -sum_k p_k log p_k``.

Modes
-----
``entropy`` (original baseline)
    R = beta * mean_b H(pi(x_b)). Rewards per-sample uniform gates; optimum pi(x) == u.
``adaptive_entropy`` (commit f296be4)
    R = mean_b sum_k beta_k * (-pi_k log pi_k), beta_k = beta / clip(pbar_k / u, clip, 1).
``regional_variance``
    R = beta * mean_k Var_b[pi_k]. Rewards different gates for different inputs;
    with pbar = u the optimum is one-hot routing (Var = 2/9 per expert for K=3).
``mutual_info``
    R = beta * [H(pbar) - mean_b H(pi(x_b))] = beta * I_hat(x; expert).
    Confident per sample, balanced in aggregate; bounded by log K.
"""

from __future__ import annotations

from typing import Callable

import torch

LOG_EPS = 1e-8  # matches the original runner implementation


def categorical_entropy(p: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """H(p) = -sum p log(p + eps) along ``dim``."""
    return -torch.sum(p * torch.log(p + LOG_EPS), dim=dim)


def entropy_reward(pi: torch.Tensor, beta: float, **_) -> torch.Tensor:
    return beta * categorical_entropy(pi).mean()


def adaptive_entropy_reward(
    pi: torch.Tensor, beta: float, adaptive_entropy_clip: float = 0.1, **_
) -> torch.Tensor:
    # Fixed gradient loophole by detaching f_m.
    f_m = pi.mean(dim=0).detach()
    target_f_m = 1.0 / pi.shape[-1]
    utilization_ratio = (f_m / target_f_m).clamp(min=adaptive_entropy_clip, max=1.0)
    per_expert_beta = beta / utilization_ratio
    per_sample = -torch.sum(
        per_expert_beta.unsqueeze(0) * pi * torch.log(pi + LOG_EPS), dim=-1
    )
    return per_sample.mean()


def regional_variance_reward(pi: torch.Tensor, beta: float, **_) -> torch.Tensor:
    return beta * torch.var(pi, dim=0, unbiased=False).mean()


def mutual_info_reward(pi: torch.Tensor, beta: float, **_) -> torch.Tensor:
    return beta * (categorical_entropy(pi.mean(dim=0)) - categorical_entropy(pi).mean())


GATE_REGULARIZERS: dict[str, Callable[..., torch.Tensor]] = {
    "entropy": entropy_reward,
    "adaptive_entropy": adaptive_entropy_reward,
    "regional_variance": regional_variance_reward,
    "mutual_info": mutual_info_reward,
}


def gate_reward(mode: str, pi: torch.Tensor, beta: float, **options) -> torch.Tensor:
    """Dispatch to the regulariser named ``mode``; returns the reward R (subtracted from the loss)."""
    try:
        fn = GATE_REGULARIZERS[mode]
    except KeyError as exc:
        raise ValueError(
            f"Unknown gate_reg_mode={mode!r}; choose from {sorted(GATE_REGULARIZERS)}"
        ) from exc
    return fn(pi, float(beta), **options)


def balance_loss(pi: torch.Tensor) -> torch.Tensor:
    """L_bal = sum_k (pbar_k - 1/K)^2."""
    return torch.sum((pi.mean(dim=0) - 1.0 / pi.shape[-1]) ** 2)
