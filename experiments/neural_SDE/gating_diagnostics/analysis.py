"""Offline real-trajectory analysis: no fitting, sampling changes, or future model inputs."""

import hashlib
import json
import pickle
import types
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from amgm.models.neural_SDE.runner import NeuralSDERunner
from amgm.models.mlp import NeuralSDEMoE
from experiments.neural_SDE.fibonacci_state_detection import detect_fibonacci_interactions
from .collection import write_manifest
from .statistics import PROBS, RATIOS, window_features, validate_probabilities, distance_summary, movement_summary

EVENT_NAMES = {"pullback": "Bounce", "breakout": "Break", "hover": "Hover"}
EVENT_COLUMNS = ["segment_id", "issue_id", "origin_date", "event", "entry_index", "hover_index",
                 "exit_index", "confirmation_end_index", "output_index", "gate_index", "duration",
                 "fib_level", "pullback_stage", "selected_example", *PROBS,
                 *[c + "_entry" for c in PROBS], *[c + "_change" for c in PROBS]]


def load_moe_checkpoint(path):
    # Historical checkpoints contain concrete Linux paths. Translate only those
    # pickle classes, without monkey-patching pathlib for the rest of the process.
    class PortableUnpickler(pickle.Unpickler):
        def find_class(self, module, name):
            if module == "pathlib" and name in ("PosixPath", "WindowsPath"):
                return Path
            return super().find_class(module, name)

    portable_pickle = types.ModuleType("portable_pickle")
    portable_pickle.Unpickler = PortableUnpickler
    portable_pickle.load = pickle.load
    checkpoint = torch.load(path, map_location="cpu", weights_only=False,
                            pickle_module=portable_pickle)
    params = dict(checkpoint["hyper_parameters"])
    params["compile_model"] = False  # Offline inference; preserves saved weights and arithmetic.
    runner = NeuralSDERunner(**params)
    if not isinstance(runner.model, NeuralSDEMoE):
        raise ValueError("Gating diagnostics require a NeuralSDEMoE checkpoint")
    state = {}
    for key, value in checkpoint["state_dict"].items():
        key = key.replace("model._orig_mod.", "model.", 1)
        if key.startswith("model.backbone."):
            key = key.replace("model.backbone.", "model.encoder.", 1)
        state[key] = value
    runner.load_state_dict(state, strict=True)
    return runner.eval(), checkpoint


def load_prices(path, issue_ids):
    """Read only needed columns in chunks; deliberately bypass dataset cache-writing loaders."""
    frames = []
    for chunk in pd.read_csv(path, sep="\t", usecols=["IssueId", "Date", "ClAdjLoc"],
                             dtype={"IssueId": str}, chunksize=250000):
        frames.append(chunk[chunk.IssueId.isin(issue_ids)])
    frame = pd.concat(frames, ignore_index=True)
    frame["Date"] = pd.to_datetime(frame.Date)
    return frame.sort_values(["IssueId", "Date"])


def segment_origins(dates, lookback=252, horizon=100, start="2018-01-01", end="2020-12-31"):
    """Disjoint future transitions; adjacent segments share only the boundary price."""
    dates = pd.DatetimeIndex(dates)
    first = max(lookback - 1, int(dates.searchsorted(pd.Timestamp(start))))
    last = int(dates.searchsorted(pd.Timestamp(end), side="right")) - 1
    return range(first, max(first, last - horizon + 1), horizon)


def align_events(events, p, metadata):
    rows = []
    for event in events:
        name = EVENT_NAMES.get(event["event_type"].value)
        if name is None:
            continue
        # A confirmed outcome is labeled retrospectively, but the gate is read
        # at the original exit, not after the ten-day confirmation window.
        gate_index = event["output_index"] if name == "Hover" else event["exit_index"]
        entry = event["entry_index"]
        row = dict(metadata, event=name, gate_index=gate_index, duration=gate_index - entry,
                   selected_example=False)
        for col in ["entry_index", "hover_index", "exit_index", "confirmation_end_index",
                    "output_index", "fib_level", "pullback_stage"]:
            row[col] = event[col]
        for k, col in enumerate(PROBS):
            row[col] = p[gate_index, k]
            row[col + "_entry"] = p[entry, k]
            row[col + "_change"] = p[gate_index, k] - p[entry, k]
        rows.append(row)
    return rows


def incomplete_interactions(prices, levels, horizon):
    """Find unresolved observed entries without altering the shared event detector.

    Extending with the last observation lets the unchanged FSM reveal entries
    whose confirmation/dwell was cut off. Only original entries with outputs
    after the observed horizon are counted; padded events are never analyzed.
    """
    extended = np.concatenate([prices, np.repeat(prices[-1], 32)])
    events = detect_fibonacci_interactions(extended, levels)
    unfinished = {e["entry_index"] for e in events
                  if e["entry_index"] <= horizon and e["output_index"] > horizon}
    finished = {e["entry_index"] for e in events
                if e["is_terminal"] and e["output_index"] <= horizon}
    return len(unfinished - finished)


def collect_real_gates(runner, prices, issue_ids, lookback=252, horizon=100):
    samples, events, segments, exclusions = [], [], [], []
    for iid in issue_ids:
        group = prices[prices.IssueId == iid].sort_values("Date")
        if group.Date.duplicated().any():
            exclusions.append(dict(issue_id=iid, reason="duplicate dates; excluded entire stock"))
            continue
        values = group.ClAdjLoc.to_numpy(dtype=np.float32)
        dates = group.Date.to_numpy()
        origins = list(segment_origins(dates, lookback, horizon))
        if not origins:
            exclusions.append(dict(issue_id=iid, reason="insufficient history or complete evaluation horizon"))
        for origin in origins:
            raw = values[origin - lookback + 1:origin + horizon + 1]
            if not np.isfinite(raw).all() or (raw <= 0).any() or np.ptp(raw[:lookback]) < 1e-8:
                exclusions.append(dict(issue_id=iid, origin_date=str(dates[origin])[:10],
                                       reason="nonfinite/nonpositive prices or constant origin history"))
                continue
            windows = np.lib.stride_tricks.sliding_window_view(raw, lookback).copy()
            x, f, d, v, nearest = window_features(windows)
            with torch.inference_mode():
                p = runner(torch.from_numpy(x), torch.from_numpy(f))[2].cpu().numpy()
            validate_probabilities(p)
            sid = len(segments)
            future = values[origin:origin + horizon + 1]
            low = float(raw[:lookback].min())
            scale = float(raw[:lookback].max() - raw[:lookback].min())
            fixed_prices = (future - low) / scale
            metadata = dict(segment_id=sid, issue_id=iid, origin_date=str(dates[origin])[:10])
            detected = detect_fibonacci_interactions(fixed_prices, RATIOS)
            event_rows = align_events(detected, p, metadata)
            events.extend(event_rows)
            segments.append(dict(metadata, history_min=low, history_range=scale, horizon=horizon,
                                 timeout_count=sum(e["event_type"].value == "timeout" for e in detected),
                                 incomplete_count=incomplete_interactions(fixed_prices, RATIOS, horizon)))
            # Include boundary once in aggregate geometry; retain it in raw data
            # because event alignment and example plots need all 101 observations.
            for j in range(horizon + 1):
                row = dict(metadata, step=j, date=str(dates[origin + j])[:10], price=float(future[j]),
                           fixed_price=float(fixed_prices[j]), distance=d[j], movement=v[j],
                           nearest_level=int(nearest[j]), aggregate=j < horizon)
                row.update(zip(PROBS, p[j]))
                row.update({f"distance_{k}": f[j, k] for k in range(7)})
                samples.append(row)
    if not samples:
        raise ValueError("No complete valid 2018-2020 evaluation segments; inspect data and stock selection")
    event_frame = pd.DataFrame(events, columns=EVENT_COLUMNS)
    for name in EVENT_NAMES.values():
        subset = event_frame[event_frame.event == name].sort_values(
            ["duration", "issue_id", "origin_date", "entry_index"], kind="stable")
        if len(subset):
            # Lower median for an even event count; selection never uses probabilities.
            median_duration = subset.duration.iloc[(len(subset) - 1) // 2]
            chosen = subset[subset.duration == median_duration].index[0]
            event_frame.loc[chosen, "selected_example"] = True
    return pd.DataFrame(samples), event_frame, pd.DataFrame(segments), pd.DataFrame(
        exclusions, columns=["issue_id", "origin_date", "reason"])


def event_summary(events):
    rows = []
    for name in ["Bounce", "Break", "Hover"]:
        group = events[events.event == name]
        n, stocks = len(group), group.issue_id.nunique()
        for col in PROBS:
            rows.append(dict(event=name, expert=col, n=n, stocks=stocks,
                             supported=n >= 30 and stocks >= 5,
                             probability=group[col].mean(), entry_probability=group[col + "_entry"].mean(),
                             change=group[col + "_change"].mean()))
    return pd.DataFrame(rows)


def analyze_checkpoint(checkpoint_path, output_dir, issue_ids=None, seed=1, data_path=None, training_run=None):
    """Generate offline diagnostics into an existing run or a fresh analysis directory."""
    from .plots import render_report
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = Path(checkpoint_path).resolve()
    manifest_file = output_dir / "manifest.json"
    manifest = json.loads(manifest_file.read_text(encoding="utf-8")) if manifest_file.exists() else {}
    runner, checkpoint = load_moe_checkpoint(checkpoint_path)
    # Explicit association prevents importing unrelated training curves into an analysis.
    if training_run is not None:
        source = Path(training_run).resolve()
        if source != output_dir.resolve():
            raise ValueError("Use the training run directory as output when attaching its history")
        saved_checkpoint = manifest.get("checkpoint")
        if not saved_checkpoint or Path(saved_checkpoint).resolve() != checkpoint_path:
            raise ValueError("Checkpoint does not match this training run's best checkpoint")
    elif manifest.get("training_history"):
        raise ValueError("Specify --training-run to attach training history; otherwise use a fresh output directory")
    if issue_ids is None:
        issue_ids = manifest.get("issue_ids") or checkpoint["hyper_parameters"]["dset_cfg"].get("issue_ids")
    if not issue_ids:
        raise ValueError("This checkpoint does not record randomly selected stocks. Supply --issue-ids-file or --training-run")
    issue_ids = sorted(set(str(i) for i in issue_ids))
    lookback = int(runner.dset_cfg["lookback_window"])
    if lookback != 252:
        raise ValueError("This experiment specifies a 252-observation lookback; checkpoint differs")
    if data_path is None:
        from amgm import config
        data_path = config.am_dataset_dir / "security_data.txt"
    manifest.update(dict(
        checkpoint=str(checkpoint_path), checkpoint_sha256=hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
        checkpoint_epoch=checkpoint.get("epoch"), issue_ids=issue_ids, analysis_seed=seed,
        configuration=manifest.get("configuration", checkpoint["hyper_parameters"]),
        training_history=bool(training_run),
        experiment1_status=manifest.get("experiment1_status", "unavailable: cannot reconstruct training history from a checkpoint"),
        evaluation=dict(start="2018-01-01", end="2020-12-31", lookback=252, horizon=100,
                        bootstrap=1000, bootstrap_unit="stock", confidence="pointwise 95%",
                        min_windows=30, min_stocks=5, bins=20,
                        detector=dict(tolerance=.02, min_dwell_steps=3, confirmation_steps=10, max_dwell_steps=20)),
        data_source=str(Path(data_path).resolve()),
    ))
    print("Reading evaluation prices and computing real-trajectory gates...", flush=True)
    prices = load_prices(data_path, issue_ids)
    samples, events, segments, exclusions = collect_real_gates(runner, prices, issue_ids)
    aggregate = samples[samples["aggregate"]]
    print(f"Analyzing {len(aggregate)} windows, {aggregate.issue_id.nunique()} stocks, {len(events)} events...", flush=True)
    distances = distance_summary(aggregate, seed=seed)
    movement = movement_summary(aggregate)
    event_metrics = event_summary(events)
    for name, frame in [("real_gates", samples), ("events", events), ("segments", segments),
                        ("exclusions", exclusions), ("distance_bins", distances),
                        ("movement_bins", movement), ("event_summary", event_metrics)]:
        frame.to_csv(output_dir / f"{name}.csv", index=False)
    np.savez_compressed(output_dir / "real_windows.npz", probabilities=samples[list(PROBS)].to_numpy(),
                        distances=samples[[f"distance_{k}" for k in range(7)]].to_numpy(),
                        issue_id=samples.issue_id.to_numpy(dtype=str), date=samples.date.to_numpy(dtype=str),
                        segment_id=samples.segment_id.to_numpy(), step=samples.step.to_numpy())
    manifest["evaluation_counts"] = dict(windows=len(aggregate), stocks=aggregate.issue_id.nunique(),
                                          segments=len(segments), events=len(events), exclusions=len(exclusions),
                                          timeouts=int(segments.timeout_count.sum()),
                                          incomplete=int(segments.incomplete_count.sum()))
    write_manifest(manifest_file, manifest)
    render_report(output_dir)
    return output_dir / "report.html"
