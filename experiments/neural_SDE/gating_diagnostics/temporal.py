"""Paired temporal evaluation, with the existing rollout injected by the CLI."""

import gzip
import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .analysis import load_moe_checkpoint, load_prices
from .collection import write_manifest
from .statistics import PROBS, window_features
from .temporal_statistics import (states_and_flags, invalid_reason, trajectory_metrics, paired_weights,
                                  compare_trajectories, dwell_tables, distribution_tables)


def origin_seed(seed, issue_id, origin_date):
    payload = f"{seed}|{issue_id}|{origin_date}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") % (2**63 - 1)


class RolloutObserver:
    """Capture the exact features passed to the unchanged generator's model calls."""
    def __init__(self, runner):
        self.runner, self.features = runner, []

    def __call__(self, x, f):
        self.features.append(f.detach().cpu().numpy().copy())
        return self.runner(x, f)


def evaluate_windows(runner, raw):
    x, f, *_ = window_features(raw)
    with torch.inference_mode():
        p = runner(torch.from_numpy(x), torch.from_numpy(f))[2].cpu().numpy()
    return p, f


def paired_origin(runner, rollout, history, future, paths, seed, dt):
    """All generated outputs are retained, including invalid paths; no repairs."""
    history = np.asarray(history, dtype=np.float32)
    future = np.asarray(future, dtype=np.float32)
    if len(history) != 252 or len(future) != 101 or history[-1] != future[0]:
        raise ValueError("Expected matching 252-history and 101-price paths")
    windows = np.lib.stride_tricks.sliding_window_view(np.r_[history[:-1], future], 252).copy()
    real_p, real_f = evaluate_windows(runner, windows)
    x, *_ = window_features(history[None, :])
    low, scale = history.min(), max(float(np.ptp(history)), 1e-8)
    batch = dict(price_window=torch.from_numpy(np.repeat(x, paths, axis=0)),
                 sample_min=torch.full((paths, 1), float(low)), sample_range=torch.full((paths, 1), scale))
    observer = RolloutObserver(runner)
    prices, gates, _, _ = rollout(observer, batch, n_steps=100, dt=dt, seed=seed)
    # The generator reconstructs the raw context from its normalized input.
    # Preserve that exact context, including float32 roundoff, for the last gate.
    reconstructed = (batch["price_window"] * batch["sample_range"] + batch["sample_min"]).numpy()
    terminal_window = np.concatenate([reconstructed, prices[:, 1:]], axis=1)[:, -252:].copy()
    terminal_p, terminal_f = evaluate_windows(runner, terminal_window)
    gates = np.concatenate([gates, terminal_p[:, None, :]], axis=1)
    features = np.concatenate([np.stack(observer.features, axis=1), terminal_f[:, None, :]], axis=1)
    np.testing.assert_allclose(gates[:, 0], np.broadcast_to(real_p[0], (paths, 3)), atol=2e-6, rtol=2e-5)
    return np.concatenate([future[None, :], prices]), np.concatenate([real_p[None, :], gates]), np.concatenate([real_f[None, :], features])


def run_temporal(training_run, output_dir, rollout, paths_per_origin=100, seed=1, data_path=None):
    if paths_per_origin < 1:
        raise ValueError("paths_per_origin must be positive")
    source, root = Path(training_run).resolve(), Path(output_dir).resolve()
    manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    checkpoint = Path(manifest["checkpoint"])
    if not checkpoint.exists():
        checkpoint = source / "checkpoints" / checkpoint.name
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    if digest != manifest["checkpoint_sha256"]:
        raise ValueError("Checkpoint SHA-256 does not match the source run")
    cohort = pd.read_csv(source / "segments.csv", dtype={"issue_id": str}).sort_values(["issue_id", "origin_date"])
    reference = pd.read_csv(source / "real_gates.csv", dtype={"issue_id": str})
    if cohort.empty or cohort.segment_id.duplicated().any() or not (cohort.horizon == 100).all():
        raise ValueError("Expected a nonempty cohort of unique 100-transition segments")
    runner, _ = load_moe_checkpoint(checkpoint)
    if int(runner.dset_cfg["lookback_window"]) != 252:
        raise ValueError("Expected a 252-observation model lookback")
    before = {k: v.detach().clone() for k, v in runner.state_dict().items()}
    data_path = Path(data_path or manifest["data_source"])
    prices = load_prices(data_path, sorted(cohort.issue_id.unique()))
    root.mkdir(parents=True, exist_ok=False)
    (root / "origins").mkdir()
    chosen = cohort.iloc[np.unique(np.linspace(0, len(cohort) - 1, min(3, len(cohort)), dtype=int))].segment_id.tolist()
    out_manifest = dict(experiment=4, status="collecting", training_run=str(source), checkpoint=str(checkpoint.resolve()),
                        checkpoint_sha256=digest, checkpoint_epoch=manifest["checkpoint_epoch"], seed=seed,
                        paths_per_origin=paths_per_origin, horizon=100, observations=101, lookback=252,
                        dt=float(runner.default_dt), data_source=str(data_path.resolve()),
                        source_cohort_sha256=hashlib.sha256((source / "segments.csv").read_bytes()).hexdigest(),
                        reference_gates_sha256=hashlib.sha256((source / "real_gates.csv").read_bytes()).hexdigest(),
                        origin_count=len(cohort), stock_count=cohort.issue_id.nunique(), example_segments=chosen,
                        seed_derivation="SHA256(master seed|stock ID|origin date), first 8 bytes little-endian modulo 2^63-1",
                        bootstrap=1000, bootstrap_unit="paired stock clusters", confidence="pointwise 95%",
                        weighting="equal origin; each valid synthetic path gets 1 / valid paths at origin",
                        tie_rule="argmax first index: Bounce, Break, Hover", ambiguity_margin=.05,
                        transition_support="at least 30 raw source transitions and 5 stocks",
                        dwell_definition="observations in consecutive winning-state runs; boundary runs censored",
                        runtime=dict(python=platform.python_version(), torch=torch.__version__, numpy=np.__version__, platform=platform.platform()))
    write_manifest(root / "manifest.json", out_manifest)
    rows, dwells, coverage, exclusions, soft_changes, examples = [], [], [], [], {}, []
    max_reference_error = 0.0
    try:
        with gzip.open(root / "steps.csv.gz", "wt", encoding="utf-8", newline="") as stream:
            for ordinal, segment in enumerate(cohort.itertuples()):
                sid, iid, date = segment.segment_id, segment.issue_id, segment.origin_date
                group = prices[prices.IssueId == iid].sort_values("Date")
                if group.Date.duplicated().any():
                    raise ValueError(f"Duplicate source dates for {iid}; cohort cannot be reproduced")
                positions = np.flatnonzero(group.Date.to_numpy() == np.datetime64(date))
                if len(positions) != 1:
                    raise ValueError(f"Origin missing from source data: {iid} {date}")
                origin = int(positions[0])
                history = group.ClAdjLoc.iloc[origin - 251:origin + 1].to_numpy(dtype=np.float32)
                future = group.ClAdjLoc.iloc[origin:origin + 101].to_numpy(dtype=np.float32)
                local_seed = origin_seed(seed, iid, date)
                path_prices, gates, features = paired_origin(runner, rollout, history, future, paths_per_origin, local_seed, runner.default_dt)
                ref = reference[reference.segment_id == sid].sort_values("step")
                if len(ref) != 101 or ref.issue_id.nunique() != 1 or ref.issue_id.iloc[0] != iid:
                    raise ValueError(f"Reference cohort mismatch at segment {sid}")
                np.testing.assert_allclose(path_prices[0], ref.price, atol=1e-5, rtol=1e-6)
                np.testing.assert_allclose(gates[0], ref[list(PROBS)], atol=2e-6, rtol=2e-5)
                max_reference_error = max(max_reference_error, float(np.abs(gates[0] - ref[list(PROBS)].to_numpy()).max()))
                sources = np.array(["real"] + ["synthetic"] * paths_per_origin)
                path_ids = np.r_[-1, np.arange(paths_per_origin)]
                reasons = [invalid_reason(px, pp, ff) for px, pp, ff in zip(path_prices, gates, features)]
                valid = np.array([not reason for reason in reasons])
                state, margins, ambiguous, tied = states_and_flags(gates)
                dates = group.Date.iloc[origin:origin + 101].dt.strftime("%Y-%m-%d").to_numpy(dtype=str)
                np.savez_compressed(root / "origins" / f"segment_{sid:04d}.npz", segment_id=sid, issue_id=iid,
                                    origin_date=date, seed=local_seed, source=sources, path_id=path_ids,
                                    step=np.arange(101), date=dates, price=path_prices, probabilities=gates,
                                    features=features, state=state, winner_margin=margins, ambiguous=ambiguous,
                                    exact_tie=tied, valid=valid, invalid_reason=np.asarray(reasons))
                npaths = len(path_ids)
                change = np.concatenate([np.full((npaths, 1), np.nan), np.abs(np.diff(gates, axis=1)).mean(2)], axis=1)
                step_frame = pd.DataFrame(dict(segment_id=sid, issue_id=iid, origin_date=date,
                                               source=np.repeat(sources, 101), path_id=np.repeat(path_ids, 101),
                                               step=np.tile(np.arange(101), npaths), date=np.tile(dates, npaths),
                                               price=path_prices.ravel(), state=state.ravel(), winner_margin=margins.ravel(),
                                               ambiguous=ambiguous.ravel(), exact_tie=tied.ravel(),
                                               valid_path=np.repeat(valid, 101), soft_change=change.ravel()))
                for k, col in enumerate(PROBS):
                    step_frame[col] = gates[:, :, k].ravel()
                for k in range(7):
                    step_frame[f"distance_{k}"] = features[:, :, k].ravel()
                step_frame.to_csv(stream, index=False, header=ordinal == 0)
                if sid in chosen:
                    examples.append(step_frame[step_frame.path_id.isin([-1, 0])])
                for index, (src, path_id, reason) in enumerate(zip(sources, path_ids, reasons)):
                    identity = dict(segment_id=sid, issue_id=iid, origin_date=date, source=src, path_id=int(path_id))
                    if reason:
                        rows.append(dict(identity, valid=False, invalid_reason=reason))
                        exclusions.append(dict(identity, reason=reason))
                    else:
                        stats, runs, changes = trajectory_metrics(gates[index], identity)
                        rows.append(dict(stats, valid=True, invalid_reason=""))
                        dwells.extend(runs)
                        soft_changes[(sid, src, int(path_id))] = changes
                paired = bool(valid[0] and valid[1:].any())
                coverage.append(dict(segment_id=sid, issue_id=iid, origin_date=date, seed=local_seed,
                                     real_valid=bool(valid[0]), generated=paths_per_origin,
                                     valid_synthetic=int(valid[1:].sum()), invalid_synthetic=int((~valid[1:]).sum()), paired=paired))
                if not paired:
                    exclusions.append(dict(segment_id=sid, issue_id=iid, origin_date=date, source="paired origin", path_id=-1,
                                           reason="no valid synthetic paths" if not valid[1:].any() else "invalid real trajectory"))
                if ordinal % 10 == 0 or ordinal == len(cohort) - 1:
                    print(f"Origins {ordinal + 1}/{len(cohort)}; synthetic paths {(ordinal + 1) * paths_per_origin:,}", flush=True)
        for key, value in runner.state_dict().items():
            torch.testing.assert_close(value, before[key], rtol=0, atol=0)
        assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == digest
        trajectories = paired_weights(pd.DataFrame(rows))
        # Define metric columns even in the complete-failure case.
        from .temporal_statistics import MEAN_METRICS
        for col in MEAN_METRICS:
            if col not in trajectories:
                trajectories[col] = np.nan
        dwell_frame = pd.DataFrame(dwells, columns=["segment_id", "issue_id", "origin_date", "source", "path_id",
                                                   "expert", "start", "end", "duration", "left_censored", "right_censored", "interior"])
        dwell_frame, dwell_distribution, dwell_summary = dwell_tables(dwell_frame, trajectories)
        comparisons, transitions, counts = compare_trajectories(trajectories, seed)
        distributions = distribution_tables(trajectories, soft_changes)
        tables = dict(trajectories=trajectories, dwell_runs=dwell_frame, dwell_distribution=dwell_distribution,
                      dwell_summary=dwell_summary, comparisons=comparisons, transition_probabilities=transitions,
                      transition_counts=counts, distributions=distributions, coverage=pd.DataFrame(coverage),
                      exclusions=pd.DataFrame(exclusions, columns=["segment_id", "issue_id", "origin_date", "source", "path_id", "reason"]))
        tables["example_steps"] = pd.concat(examples, ignore_index=True)
        for name, frame in tables.items():
            frame.to_csv(root / f"{name}.csv", index=False)
        out_manifest.update(status="complete", paired_origins=int(pd.DataFrame(coverage).paired.sum()),
                            valid_synthetic=int(pd.DataFrame(coverage).valid_synthetic.sum()),
                            invalid_synthetic=int(pd.DataFrame(coverage).invalid_synthetic.sum()),
                            max_reference_gate_error=max_reference_error, checkpoint_unchanged=True)
        write_manifest(root / "manifest.json", out_manifest)
        from .temporal_plots import render_temporal_report
        render_temporal_report(root)
        return root / "report.html"
    except Exception as exc:
        out_manifest.update(status="failed", error=str(exc), completed_origins=len(coverage))
        write_manifest(root / "manifest.json", out_manifest)
        raise
