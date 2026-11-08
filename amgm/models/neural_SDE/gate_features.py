"""Gate-specific input features for the MoE neural SDE.

Every feature set maps a min-max normalised price window ``x`` of shape
``(B, W)`` (the same tensor the model receives as ``x_hist``) to causal features
``g`` of shape ``(B, G)``. Only past and current prices are used. The model
selects how the gate consumes them with ``gate_inputs``:

``h`` (original baseline)
    gate(h), h = [Z_t || f_t]. No gate features.
``features``
    gate(g). The gate sees only g; experts keep h.
``h+features``
    gate([h || g]).

Notation: ``t`` is the last index, ``lo, hi`` the window min/max,
``delta = hi - lo``, ``l_k = lo + r_k * delta`` the Fibonacci levels,
``dx_s = x_s - x_{s-1}``, ``sigma_n = std(dx over the last n days)`` and ``L*`` the
level nearest to ``x_t``. Because ``x = (P - min) / range``,
``(x_t - l_k) / sigma_20 = (P_t - L_k) / std(dP)``, which equals the raw-price
form ``(P - L_k) / (sigma20_log * P)`` up to a factor ``mean(P_20) / P_t``.

Feature sets
------------
``price_v1`` (14 features)
    dist_vol_k  = clip((x_t - l_k) / sigma_20, +-clip),            k = 0..6
    approach    = -sign(x_t - L*) * sign(x_t - x_{t-1})             (+1: moving toward L*)
    vel_k       = clip((x_t - x_{t-k}) / (sigma_20 * sqrt(k)), +-clip),  k = 1, 5, 10
    log_vol_ratio = clip(log(sigma_20 / sigma_60), +-clip)
    log_range_vol = clip(log(delta / (sigma_20 * sqrt(252))), +-clip)
    zone_days   = consecutive days ending at t with |x_s - L*| <= zone_tol * delta,
                  capped at max_dwell, divided by max_dwell. zone_tol=0.02 is the
                  tolerance of the Fibonacci event detector.
``vol_only`` (2 features)
    log_vol_ratio, log_range_vol. Ablation: does the gate only route on volatility?
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import torch

FIB_RATIOS = (0.0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0)
VELOCITY_LAGS = (1, 5, 10)
GATE_INPUT_MODES = ("h", "features", "h+features")

PRICE_V1_NAMES = (
    *[f"dist_vol_{k}" for k in range(len(FIB_RATIOS))],
    "approach",
    *[f"vel_{k}" for k in VELOCITY_LAGS],
    "log_vol_ratio",
    "log_range_vol",
    "zone_days",
)


def _std(dx: torch.Tensor, n: int, eps: float) -> torch.Tensor:
    return dx[:, -n:].std(dim=1, unbiased=False, keepdim=True).clamp_min(eps)


def price_v1(
    x: torch.Tensor,
    clip: float = 5.0,
    zone_tol: float = 0.02,
    max_dwell: int = 20,
    eps: float = 1e-6,
) -> torch.Tensor:
    if x.ndim != 2 or x.shape[1] < 61:
        raise ValueError(f"price_v1 needs a (B, W>=61) window, got {tuple(x.shape)}")
    lo = x.min(dim=1, keepdim=True).values
    delta = (x.max(dim=1, keepdim=True).values - lo).clamp_min(eps)
    ratios = torch.tensor(FIB_RATIOS, dtype=x.dtype, device=x.device)
    levels = lo + ratios * delta  # (B, 7)

    x_t = x[:, -1:]
    dx = x[:, 1:] - x[:, :-1]
    sigma20, sigma60 = _std(dx, 20, eps), _std(dx, 60, eps)

    signed = x_t - levels
    dist_vol = (signed / sigma20).clamp(-clip, clip)
    nearest = torch.gather(levels, 1, signed.abs().argmin(dim=1, keepdim=True))
    approach = -torch.sign(x_t - nearest) * torch.sign(dx[:, -1:])

    velocities = [
        ((x_t - x[:, -1 - k:-k]) / (sigma20 * math.sqrt(k))).clamp(-clip, clip)
        for k in VELOCITY_LAGS
    ]
    log_vol_ratio = torch.log(sigma20 / sigma60).clamp(-clip, clip)
    log_range_vol = torch.log(delta / (sigma20 * math.sqrt(252.0))).clamp(-clip, clip)

    tail = x[:, -max_dwell:]
    inside = ((tail - nearest).abs() <= zone_tol * delta).to(x.dtype)
    zone_days = inside.flip(1).cumprod(dim=1).sum(dim=1, keepdim=True) / max_dwell

    return torch.cat(
        [dist_vol, approach, *velocities, log_vol_ratio, log_range_vol, zone_days], dim=1
    )


_VOL_ONLY_IDX = [PRICE_V1_NAMES.index("log_vol_ratio"), PRICE_V1_NAMES.index("log_range_vol")]


def vol_only(x: torch.Tensor, **options) -> torch.Tensor:
    return price_v1(x, **options)[:, _VOL_ONLY_IDX]


@dataclass(frozen=True)
class GateFeatureSet:
    fn: Callable[..., torch.Tensor]
    names: tuple[str, ...]
    min_lookback: int


GATE_FEATURE_SETS: dict[str, GateFeatureSet] = {
    "price_v1": GateFeatureSet(price_v1, PRICE_V1_NAMES, 61),
    "vol_only": GateFeatureSet(vol_only, ("log_vol_ratio", "log_range_vol"), 61),
}


def get_gate_feature_set(name: str) -> GateFeatureSet:
    try:
        return GATE_FEATURE_SETS[name]
    except KeyError as exc:
        raise ValueError(
            f"Unknown gate_feature_set={name!r}; choose from {sorted(GATE_FEATURE_SETS)}"
        ) from exc


def gate_feature_names(name: str) -> tuple[str, ...]:
    return get_gate_feature_set(name).names


def compute_gate_features(name: str, x_hist: torch.Tensor, **options) -> torch.Tensor:
    """Dispatch to the feature set named ``name``; returns ``(B, G)`` gate features."""
    return get_gate_feature_set(name).fn(x_hist, **options)
