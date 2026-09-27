"""Pure numerical helpers. All randomness is local to the analysis."""

import numpy as np
import pandas as pd

EXPERTS = ("Bounce", "Break", "Hover")
PROBS = ("p_bounce", "p_break", "p_hover")
RATIOS = np.array([0, .236, .382, .5, .618, .786, 1.0], dtype=np.float32)


def validate_probabilities(p):
    p = np.asarray(p, dtype=np.float64)
    if p.ndim != 2 or p.shape[1] != 3 or not len(p):
        raise ValueError("Expected a nonempty (N, 3) MoE probability array")
    if not np.isfinite(p).all() or (p < 0).any() or not np.allclose(p.sum(1), 1, atol=1e-5):
        raise ValueError("Invalid MoE probabilities; MLP/PatchTST placeholder gates are unsupported")
    return p


class GateAccumulator:
    """Sufficient statistics avoid retaining batch graphs or weighting small batches equally."""

    def __init__(self):
        self.n = 0
        self.total = np.zeros(3)
        self.square = np.zeros(3)
        self.wins = np.zeros(3)
        self.entropy = self.confidence = 0.0

    def update(self, p):
        p = validate_probabilities(p)
        self.n += len(p)
        self.total += p.sum(0)
        self.square += (p * p).sum(0)
        tied = p == p.max(1, keepdims=True)
        self.wins += (tied / tied.sum(1, keepdims=True)).sum(0)
        self.entropy += float(-(p * np.log(np.maximum(p, 1e-300))).sum() / np.log(3))
        self.confidence += float(p.max(1).sum())

    def summary(self):
        if not self.n:
            raise ValueError("No observations collected")
        mean = self.total / self.n
        result = dict(n=self.n, entropy=self.entropy / self.n, confidence=self.confidence / self.n)
        for k, col in enumerate(PROBS):
            result[col] = mean[k]
            result[col + "_variance"] = max(0.0, self.square[k] / self.n - mean[k] ** 2)
            result[col + "_winner_share"] = self.wins[k] / self.n
        return result


def assignment_change(previous, current, margin=.05):
    previous, current = validate_probabilities(previous), validate_probabilities(current)
    if previous.shape != current.shape:
        raise ValueError("Validation cohorts differ")
    a, b = np.sort(previous, axis=1), np.sort(current, axis=1)
    keep = (a[:, -1] - a[:, -2] >= margin) & (b[:, -1] - b[:, -2] >= margin)
    return dict(
        n=len(current), eligible_n=int(keep.sum()), exclusion_rate=float(1 - keep.mean()),
        switch_fraction=float((previous.argmax(1)[keep] != current.argmax(1)[keep]).mean()) if keep.any() else np.nan,
        mean_absolute_probability_change=float(np.abs(current - previous).mean()),
    )


def window_features(windows):
    """Mirror dataset min-max normalization and Fibonacci arithmetic, without future values."""
    windows = np.asarray(windows, dtype=np.float32)
    minimum = windows.min(1, keepdims=True)
    scale = np.maximum(windows.max(1, keepdims=True) - minimum, 1e-8)
    normalized = (windows - minimum) / scale
    low, high = normalized.min(1, keepdims=True), normalized.max(1, keepdims=True)
    delta = high - low
    levels = np.concatenate([low + r * delta for r in RATIOS], axis=1).astype(np.float32)
    features = (normalized[:, -1:] - levels) / np.maximum(delta, 1e-8)
    nearest = np.abs(features).argmin(1)
    distance = features[np.arange(len(features)), nearest]
    movement = normalized[:, -1] - normalized[:, -2]
    return normalized, features, distance, movement, nearest


def equal_edges(values, bins=20):
    values = np.asarray(values, dtype=float)
    if not len(values) or not np.isfinite(values).all():
        raise ValueError("Binning requires finite observations")
    low, high = values.min(), values.max()
    if low == high:
        pad = max(abs(low) * 1e-6, 1e-8)
        low, high = low - pad, high + pad
    return np.linspace(low, high, bins + 1)


def bin_indices(values, edges):
    return np.clip(np.searchsorted(edges, values, side="right") - 1, 0, len(edges) - 2)


def distance_summary(frame, bootstrap=1000, seed=1, min_windows=30, min_stocks=5):
    """Pointwise stock-cluster percentile bands, using pooled-window estimands.

    The same resampled stock weights apply to all bins. Missing bootstrap bins
    are excluded from their interval and the valid replicate count is exported.
    """
    edges = equal_edges(frame.distance)
    stocks = sorted(frame.issue_id.unique())
    stock_index = {s: i for i, s in enumerate(stocks)}
    rng = np.random.default_rng(seed)
    weights = rng.multinomial(len(stocks), np.full(len(stocks), 1 / len(stocks)), size=bootstrap)
    rows = []
    for level in [-1, *range(7)]:
        subset = frame if level == -1 else frame[frame.nearest_level == level]
        count = np.zeros((len(stocks), 20))
        sums = np.zeros((len(stocks), 20, 3))
        si = np.array([stock_index[s] for s in subset.issue_id], dtype=int)
        bi = bin_indices(subset.distance.to_numpy(), edges)
        np.add.at(count, (si, bi), 1)
        np.add.at(sums, (si, bi), subset[list(PROBS)].to_numpy())
        boot_count = weights @ count
        for j in range(20):
            n, ns = int(count[:, j].sum()), int((count[:, j] > 0).sum())
            supported = n >= min_windows and ns >= min_stocks
            row = dict(level=level, bin=j, left=edges[j], right=edges[j + 1],
                       center=(edges[j] + edges[j + 1]) / 2, n=n, stocks=ns, supported=supported)
            valid = boot_count[:, j] > 0
            row["bootstrap_valid"] = int(valid.sum())
            for k, col in enumerate(PROBS):
                row[col] = sums[:, j, k].sum() / n if n else np.nan
                draws = (weights[valid] @ sums[:, j, k]) / boot_count[valid, j]
                interval = np.quantile(draws, [.025, .975]) if supported and len(draws) else [np.nan, np.nan]
                row[col + "_low"], row[col + "_high"] = interval
            rows.append(row)
    return pd.DataFrame(rows)


def movement_summary(frame, min_windows=30, min_stocks=5):
    de, me = equal_edges(frame.distance), equal_edges(frame.movement)
    work = frame.copy()
    work["distance_bin"] = bin_indices(work.distance, de)
    work["movement_bin"] = bin_indices(work.movement, me)
    groups = {(i, j): g for (i, j), g in work.groupby(["distance_bin", "movement_bin"])}
    rows = []
    for i in range(20):
        for j in range(20):
            g = groups.get((i, j))
            n, ns = (len(g), g.issue_id.nunique()) if g is not None else (0, 0)
            row = dict(distance_bin=i, movement_bin=j, distance_left=de[i], distance_right=de[i + 1],
                       movement_left=me[j], movement_right=me[j + 1], n=n, stocks=ns,
                       supported=n >= min_windows and ns >= min_stocks)
            for col in PROBS:
                row[col] = g[col].mean() if n else np.nan
            rows.append(row)
    return pd.DataFrame(rows)
