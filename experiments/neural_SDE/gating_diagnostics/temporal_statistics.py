"""Trajectory-local temporal metrics and paired, stock-cluster comparisons."""

import numpy as np
import pandas as pd

from .statistics import EXPERTS, PROBS, validate_probabilities

MEAN_METRICS = ["switch_count", "switches_per_100", "zero_switch", "dwell_mean", "dwell_median",
                "soft_change_mean", "soft_change_median", "entropy", "ambiguous_fraction", "tie_fraction",
                *PROBS, *[p + "_winner_share" for p in PROBS]]


def states_and_flags(p):
    p = np.asarray(p)
    finite = np.isfinite(p).all(-1)
    state = np.where(finite, np.argmax(p, axis=-1), -1)
    ordered = np.sort(p, axis=-1)
    margin = ordered[..., -1] - ordered[..., -2]
    return state, margin, finite & (margin < .05), finite & (margin == 0)


def invalid_reason(prices, probabilities, features):
    if not np.isfinite(prices).all():
        return "nonfinite price"
    if (np.asarray(prices) <= 0).any():
        return "nonpositive price"
    if not np.isfinite(features).all():
        return "nonfinite Fibonacci features"
    try:
        validate_probabilities(probabilities)
    except ValueError:
        return "invalid gate probabilities"
    return ""


def trajectory_metrics(p, identity):
    p = validate_probabilities(p)
    if len(p) < 2:
        raise ValueError("Temporal metrics require at least two observations")
    state, margin, ambiguous, tied = states_and_flags(p)
    changes = np.abs(np.diff(p, axis=0)).mean(1)
    starts = np.r_[0, np.flatnonzero(state[1:] != state[:-1]) + 1]
    ends = np.r_[starts[1:] - 1, len(state) - 1]
    dwell = [dict(identity, expert=EXPERTS[state[a]], start=int(a), end=int(b),
                  duration=int(b - a + 1), left_censored=bool(a == 0),
                  right_censored=bool(b == len(state) - 1), interior=bool(a > 0 and b < len(state) - 1))
             for a, b in zip(starts, ends)]
    durations = ends - starts + 1
    count = np.zeros((3, 3), dtype=int)
    np.add.at(count, (state[:-1], state[1:]), 1)
    row = dict(identity, observations=len(p), transitions=len(p) - 1, switch_count=len(starts) - 1,
               switches_per_100=100 * (len(starts) - 1) / (len(p) - 1), zero_switch=int(len(starts) == 1),
               dwell_mean=float(durations.mean()), dwell_median=float(np.median(durations)),
               soft_change_mean=float(changes.mean()), soft_change_median=float(np.median(changes)),
               entropy=float(-(p * np.log(np.maximum(p, 1e-300))).sum(1).mean() / np.log(3)),
               ambiguous_fraction=float(ambiguous.mean()), tie_fraction=float(tied.mean()))
    for k, col in enumerate(PROBS):
        row[col] = float(p[:, k].mean())
        row[col + "_winner_share"] = float((state == k).mean())
        dd = [d["duration"] for d in dwell if d["expert"] == EXPERTS[k]]
        row[col + "_dwell_mean"] = float(np.mean(dd)) if dd else np.nan
        row[col + "_dwell_median"] = float(np.median(dd)) if dd else np.nan
        for j in range(3):
            row[f"count_{k}_{j}"] = int(count[k, j])
            row[f"transition_{k}_{j}"] = float(count[k, j] / count[k].sum()) if count[k].sum() else np.nan
    return row, dwell, changes


def paired_weights(trajectories):
    frame = trajectories.copy()
    valid = frame[frame.valid]
    good = set(valid[valid.source == "real"].segment_id) & set(valid[valid.source == "synthetic"].segment_id)
    frame["included"] = frame.valid & frame.segment_id.isin(good)
    frame["weight"] = 0.0
    use = frame[frame.included]
    if len(use):
        size = use.groupby(["segment_id", "source"]).path_id.transform("size")
        frame.loc[use.index, "weight"] = 1 / size
    return frame


def weighted_quantile(values, weights, q=.5):
    values, weights = np.asarray(values), np.asarray(weights)
    keep = np.isfinite(values) & (weights > 0)
    if not keep.any():
        return np.nan
    order = np.argsort(values[keep], kind="stable")
    x, w = values[keep][order], weights[keep][order]
    return float(x[np.searchsorted(np.cumsum(w), q * w.sum(), side="left")])


def _interval(draws):
    draws = np.asarray(draws)
    draws = draws[np.isfinite(draws)]
    return (*np.quantile(draws, [.025, .975]), len(draws)) if len(draws) else (np.nan, np.nan, 0)


def compare_trajectories(trajectories, seed=1, bootstrap=1000):
    """Origin-weighted means; bootstrap resamples the same stocks on both sides."""
    use = trajectories[trajectories.included].copy()
    if use.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    stocks = sorted(use.issue_id.unique())
    rng = np.random.default_rng(seed)
    resamples = rng.multinomial(len(stocks), np.full(len(stocks), 1 / len(stocks)), size=bootstrap)
    summaries, transitions, origin_counts = [], [], []
    draws_by_source = {}
    for source in ["real", "synthetic"]:
        g = use[use.source == source]
        denominator = g.groupby("issue_id").weight.sum().reindex(stocks).to_numpy()
        denom_draw = resamples @ denominator
        draws_by_source[source] = {}
        for metric in MEAN_METRICS:
            sums = (g[metric] * g.weight).groupby(g.issue_id).sum().reindex(stocks).to_numpy()
            mean = float(sums.sum() / denominator.sum())
            draws = (resamples @ sums) / denom_draw
            lo, hi, nboot = _interval(draws)
            draws_by_source[source][metric] = draws
            summaries.append(dict(source=source, metric=metric, mean=mean,
                                  median=weighted_quantile(g[metric], g.weight), low=lo, high=hi,
                                  bootstrap_valid=nboot, trajectories=len(g), origins=g.segment_id.nunique(), stocks=len(stocks)))
        for i in range(3):
            cols = [f"count_{i}_{j}" for j in range(3)]
            stock_counts = g[cols].multiply(g.weight, axis=0).groupby(g.issue_id).sum().reindex(stocks)
            denominator = stock_counts.sum(axis=1).to_numpy()
            source_stocks = int((denominator > 0).sum())
            raw_count = int(g[cols].sum().sum())
            supported = raw_count >= 30 and source_stocks >= 5
            den_draw = resamples @ denominator
            for j, col in enumerate(cols):
                counts = stock_counts[col].to_numpy()
                draws = np.divide(resamples @ counts, den_draw, out=np.full(bootstrap, np.nan), where=den_draw > 0)
                lo, hi, nb = _interval(draws)
                probability = counts.sum() / denominator.sum() if denominator.sum() else np.nan
                transitions.append(dict(source=source, from_expert=EXPERTS[i], to_expert=EXPERTS[j],
                                        probability=probability, weighted_count=counts.sum(), raw_source_count=raw_count,
                                        weighted_source_count=denominator.sum(), stocks=source_stocks, supported=supported,
                                        low=lo if supported else np.nan, high=hi if supported else np.nan, bootstrap_valid=nb))
                draws_by_source[source][f"transition_{i}_{j}"] = draws
        for (sid, iid), sub in g.groupby(["segment_id", "issue_id"]):
            for i in range(3):
                for j in range(3):
                    col = f"count_{i}_{j}"
                    origin_counts.append(dict(segment_id=sid, issue_id=iid, source=source,
                                              from_expert=EXPERTS[i], to_expert=EXPERTS[j],
                                              raw_count=sub[col].sum(), weighted_count=(sub[col] * sub.weight).sum()))
    for metric in MEAN_METRICS:
        a, b = [next(r for r in summaries if r["source"] == src and r["metric"] == metric) for src in ["real", "synthetic"]]
        lo, hi, nb = _interval(draws_by_source["synthetic"][metric] - draws_by_source["real"][metric])
        summaries.append(dict(source="synthetic-minus-real", metric=metric, mean=b["mean"] - a["mean"],
                              median=np.nan, low=lo, high=hi, bootstrap_valid=nb, trajectories=np.nan,
                              origins=a["origins"], stocks=len(stocks)))
    for i in range(3):
        for j in range(3):
            a, b = [next(r for r in transitions if r["source"] == src and r["from_expert"] == EXPERTS[i]
                         and r["to_expert"] == EXPERTS[j]) for src in ["real", "synthetic"]]
            supported = a["supported"] and b["supported"]
            key = f"transition_{i}_{j}"
            lo, hi, nb = _interval(draws_by_source["synthetic"][key] - draws_by_source["real"][key])
            transitions.append(dict(source="synthetic-minus-real", from_expert=EXPERTS[i], to_expert=EXPERTS[j],
                                    probability=b["probability"] - a["probability"], supported=supported,
                                    low=lo if supported else np.nan, high=hi if supported else np.nan,
                                    bootstrap_valid=nb, stocks=min(a["stocks"], b["stocks"])))
    return pd.DataFrame(summaries), pd.DataFrame(transitions), pd.DataFrame(origin_counts)


def dwell_tables(dwells, trajectories):
    keys = ["segment_id", "source", "path_id"]
    frame = dwells.merge(trajectories[keys + ["weight", "included"]], on=keys, validate="many_to_one")
    rows, summary = [], []
    for source in ["real", "synthetic"]:
        for expert in EXPERTS:
            group = frame[(frame.source == source) & (frame.expert == expert) & frame.included]
            total = group.weight.sum()
            for interior in [False, True]:
                part = group[group.interior.astype(bool) == interior]
                for duration in range(1, 102):
                    mass = part.loc[part.duration == duration, "weight"].sum()
                    rows.append(dict(source=source, expert=expert, interior=interior, duration=duration,
                                     weighted_count=mass, probability=mass / total if total else np.nan))
            for kind, part in [("all observed", group), ("fully observed interior", group[group.interior.astype(bool)])]:
                w = part.weight.sum()
                summary.append(dict(source=source, expert=expert, kind=kind, runs=len(part),
                                    weighted_runs=w, mean=(part.duration * part.weight).sum() / w if w else np.nan,
                                    median=weighted_quantile(part.duration, part.weight),
                                    censored_fraction=(part.loc[~part.interior.astype(bool), "weight"].sum() / w) if w else np.nan))
    return frame, pd.DataFrame(rows), pd.DataFrame(summary)


def distribution_tables(trajectories, soft_changes):
    """Store every plotted bin; step changes are supplied as arrays per valid path."""
    use = trajectories[trajectories.included]
    rows = []
    for metric in ["switch_count", "soft_change_mean", "step_soft_change"]:
        values = np.concatenate([np.asarray(soft_changes[(r.segment_id, r.source, r.path_id)])
                                 for r in use.itertuples()]) if metric == "step_soft_change" and len(use) else use.get(metric, pd.Series(dtype=float)).to_numpy()
        maximum = float(values.max()) if len(values) else 0.0
        edges = np.arange(-.5, 101.5, 1) if metric == "switch_count" else np.linspace(0, max(maximum, 1e-8), 41)
        for source in ["real", "synthetic"]:
            g = use[use.source == source]
            if metric == "step_soft_change":
                arrays = [soft_changes[(r.segment_id, r.source, r.path_id)] for r in g.itertuples()]
                x = np.concatenate(arrays) if arrays else np.array([])
                w = np.repeat(g.weight.to_numpy(), 100) / 100
            else:
                x, w = g.get(metric, pd.Series(dtype=float)).to_numpy(), g.weight.to_numpy()
            hist, _ = np.histogram(x, edges, weights=w)
            for j, mass in enumerate(hist):
                rows.append(dict(metric=metric, source=source, bin=j, left=edges[j], right=edges[j + 1],
                                 center=(edges[j] + edges[j + 1]) / 2, weighted_count=mass,
                                 probability=mass / hist.sum() if hist.sum() else np.nan))
    return pd.DataFrame(rows)
