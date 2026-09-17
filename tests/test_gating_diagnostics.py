"""Numerical and integration tests for observational gate diagnostics."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from lightning import Callback, Trainer, seed_everything
from torch.utils.data import DataLoader, Dataset, Subset

from amgm.data.neural_SDE import NeuralSDESample
from amgm.models.mlp import NeuralSDEMoE
from amgm.models.neural_SDE.runner import NeuralSDERunner
from amgm.utils.common import calculate_fibLevels
from experiments.neural_SDE.fibonacci_state_detection import detect_fibonacci_interactions
from experiments.neural_SDE.gating_diagnostics.analysis import (
    align_events, collect_real_gates, event_summary, incomplete_interactions,
    load_moe_checkpoint, segment_origins,
)
from experiments.neural_SDE.gating_diagnostics.collection import GatingDiagnostics
from experiments.neural_SDE.gating_diagnostics.statistics import (
    GateAccumulator, PROBS, assignment_change, distance_summary, movement_summary,
    validate_probabilities, window_features,
)


def test_weighted_population_stats_and_ties():
    p = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [.5, .5, 0], [1 / 3] * 3])
    acc = GateAccumulator()
    acc.update(p[:1])
    acc.update(p[1:])
    out = acc.summary()
    np.testing.assert_allclose([out[c] for c in PROBS], p.mean(0))
    np.testing.assert_allclose([out[c + "_variance"] for c in PROBS], p.var(0))
    np.testing.assert_allclose([out[c + "_winner_share"] for c in PROBS], [1.833333 / 5, 1.833333 / 5, 1.333333 / 5], atol=1e-6)
    for data, entropy in [(np.ones((4, 3)) / 3, 1), (np.tile([1, 0, 0], (4, 1)), 0)]:
        acc = GateAccumulator()
        acc.update(data)
        assert acc.summary()["entropy"] == pytest.approx(entropy)
    with pytest.raises(ValueError):
        validate_probabilities(np.zeros((2, 3)))


def test_ambiguous_winners_excluded_but_probability_changes_retained():
    a = np.array([[.8, .1, .1], [.34, .33, .33], [.1, .8, .1]])
    b = np.array([[.1, .8, .1], [.33, .34, .33], [.1, .8, .1]])
    stats = assignment_change(a, b)
    assert stats["switch_fraction"] == .5
    assert stats["exclusion_rate"] == pytest.approx(1 / 3)
    assert stats["mean_absolute_probability_change"] == pytest.approx(np.abs(a - b).mean())
    assert np.isnan(assignment_change(a[1:2], b[1:2])["switch_fraction"])


def test_features_match_training_and_nearest_level_rule():
    raw = np.array([[10, 14, 11, 12.1], [10, 14, 12, 11.8], [7, 7, 7, 7]], dtype=np.float32)
    x, f, d, movement, nearest = window_features(raw)
    levels, scale = calculate_fibLevels(x)
    np.testing.assert_allclose(f, (x[:, -1:] - levels) / np.maximum(scale, 1e-8), atol=1e-7)
    np.testing.assert_allclose(movement, (raw[:, -1] - raw[:, -2]) / np.maximum(np.ptp(raw, axis=1), 1e-8))
    assert d[0] > 0 and d[1] < 0
    assert nearest[0] == nearest[1] == 3
    assert nearest[2] == 0  # exact tie matches torch.argmin's first index
    _, changed, changed_distance, _, changed_nearest = window_features([[10, 14, 11, 11.6]])
    assert changed_nearest[0] == 2 != nearest[0]
    assert changed_distance[0] == changed[0, 2]


def frame_for_bins():
    rows = []
    for i in range(6):
        for j in range(100):
            p = [.2 + i * .1, .6 - i * .1, .2]
            rows.append(dict(issue_id=f"{i:08d}", distance=j / 1000 - .05,
                             movement=(j % 10) / 100 - .04, nearest_level=3, **dict(zip(PROBS, p))))
    return pd.DataFrame(rows)


def test_stock_bootstrap_counts_empty_bins_and_rng_isolation():
    frame = frame_for_bins()
    before = np.random.get_state()
    table = distance_summary(frame, bootstrap=100, seed=2)
    after = np.random.get_state()
    np.testing.assert_array_equal(before[1], after[1])
    pooled = table[table.level == -1]
    assert pooled.n.sum() == len(frame)
    assert pooled.supported.all()
    assert (pooled.p_bounce_low < pooled.p_bounce_high).all()
    assert not table[table.level == 0].supported.any()
    assert table[table.level == 0].p_bounce.isna().all()
    pd.testing.assert_frame_equal(table, distance_summary(frame, bootstrap=100, seed=2))
    sparse = distance_summary(frame[frame.issue_id == "00000000"], bootstrap=20)
    assert not sparse.supported.any()
    grid = movement_summary(frame)
    assert len(grid) == 400 and grid.n.sum() == len(frame)


def test_event_measurement_is_at_exit_not_confirmation():
    prices = np.array([.45, .50, .501, .499] + [.54] * 12)
    events = detect_fibonacci_interactions(prices, [.5])
    p = np.column_stack([np.linspace(.1, .6, len(prices)), np.linspace(.6, .1, len(prices)), np.full(len(prices), .3)])
    rows = align_events(events, p, dict(segment_id=0, issue_id="a", origin_date="2018-01-01"))
    hover, breakout = rows
    assert hover["gate_index"] == 3
    assert breakout["gate_index"] == 4 and breakout["confirmation_end_index"] == 13
    assert breakout["p_break"] == p[4, 1]
    assert breakout["p_break_entry"] == p[1, 1]
    assert incomplete_interactions(prices[:7], np.array([.5]), 6) == 1
    assert incomplete_interactions(prices, np.array([.5]), len(prices) - 1) == 0


def test_segments_are_nonoverlapping_and_within_evaluation_dates():
    dates = pd.bdate_range("2016-01-01", "2021-12-31")
    origins = list(segment_origins(dates))
    assert len(origins) > 1
    assert all(b - a == 100 for a, b in zip(origins, origins[1:]))
    assert dates[origins[0]] >= pd.Timestamp("2018-01-01")
    assert dates[origins[-1] + 100] <= pd.Timestamp("2020-12-31")
    assert all(o >= 251 for o in origins)


class CaptureRunner:
    def __init__(self):
        self.windows = []

    def __call__(self, x, f):
        self.windows.append(x.clone())
        p = torch.softmax(torch.stack([x[:, -1], f[:, 3], -x[:, -1]], 1), 1)
        return None, None, p


def test_no_future_inputs_and_deterministic_example_selection():
    dates = pd.bdate_range("2017-01-01", "2019-01-01")
    values = 100 + np.sin(np.arange(len(dates)) / 5) * 5 + np.arange(len(dates)) / 100
    prices = pd.DataFrame(dict(IssueId="a", Date=dates, ClAdjLoc=values))
    a, b = CaptureRunner(), CaptureRunner()
    samples, events, segments, _ = collect_real_gates(a, prices, ["a"])
    origin = list(segment_origins(dates))[0]
    altered = prices.copy()
    altered.loc[origin + 50:, "ClAdjLoc"] += 10
    changed, _, _, _ = collect_real_gates(b, altered, ["a"])
    torch.testing.assert_close(a.windows[0][:50], b.windows[0][:50], rtol=0, atol=0)
    np.testing.assert_array_equal(samples[list(PROBS)].to_numpy()[:50], changed[list(PROBS)].to_numpy()[:50])
    for name, group in events.groupby("event"):
        selected = group[group.selected_example]
        assert len(selected) == 1
        assert selected.duration.iloc[0] == sorted(group.duration)[(len(group) - 1) // 2]
        tied = group[group.duration == selected.duration.iloc[0]].sort_values(
            ["issue_id", "origin_date", "entry_index"])
        assert selected.index[0] == tied.index[0]
    assert len(event_summary(events)) == 9


class TinyDataset(Dataset):
    def __init__(self):
        g = torch.Generator().manual_seed(87)
        raw = torch.randn(37, 12, generator=g).numpy()
        self.x, self.f, *_ = window_features(raw)
        self.issue_ids = [f"stock{i % 5}" for i in range(37)]
        self.test_dates = [f"2017-01-{i + 1:02d}" for i in range(37)]

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        return NeuralSDESample(torch.tensor(self.x[i]), torch.tensor([self.x[i, -1] + .01]),
                               torch.tensor([0.]), torch.tensor([1.]), torch.tensor(self.f[i]),
                               torch.arange(7).float(), self.test_dates[i], self.issue_ids[i])


class Trace(Callback):
    def __init__(self):
        self.rows, self.gates = [], []

    def on_fit_start(self, trainer, module):
        self.handle = module.model.register_forward_hook(
            lambda m, inputs, output: self.gates.append(output[2].detach().clone()))

    def on_train_batch_end(self, trainer, module, outputs, batch, batch_idx):
        moments = [value.detach().clone() for state in trainer.optimizers[0].state.values()
                   for value in state.values() if isinstance(value, torch.Tensor)]
        self.rows.append((list(batch.issue_ids), outputs["loss"].detach().clone(),
                          [p.detach().clone() for p in module.parameters()],
                          [p.grad.detach().clone() for p in module.parameters()], moments))

    def on_fit_end(self, trainer, module):
        self.handle.remove()


def runner_config():
    return dict(run_cfg={}, dset_cfg={"dt": 1., "lookback_window": 12},
                model_cfg=dict(_target_=NeuralSDEMoE, lookback_window=12, num_features=7, hidden_sizes=[8, 4]),
                loss_cfg=dict(_target_=torch.nn.MSELoss), acc_cfg={},
                optim_cfg=dict(_target_=torch.optim.Adam, lr=.001), sched_cfg=None,
                compile_model=False)


@pytest.mark.parametrize("compiled", [False, True])
def test_observer_leaves_training_unchanged_and_cleans_hooks(tmp_path, compiled):
    torch.set_num_threads(1)

    def run(enabled):
        seed_everything(1, workers=True)
        dataset = TinyDataset()
        train, val = Subset(dataset, list(range(27))), Subset(dataset, list(range(27, 37)))
        runner = NeuralSDERunner(**runner_config())
        if compiled:
            runner.model = torch.compile(runner.model, backend="eager")
        trace = Trace()
        diagnostic = GatingDiagnostics(tmp_path / f"run_{compiled}", 1, runner_config(), train, val) if enabled else None
        trainer = Trainer(max_epochs=2, logger=False, enable_checkpointing=False, enable_progress_bar=False,
                          enable_model_summary=False, callbacks=[trace] + ([diagnostic] if enabled else []))
        trainer.fit(runner, train_dataloaders=DataLoader(train, batch_size=8, shuffle=True),
                    val_dataloaders=DataLoader(val, batch_size=6))
        rng_state = torch.get_rng_state().clone()
        if diagnostic:
            trainer.validate(runner, dataloaders=DataLoader(val, batch_size=6), verbose=False)
            assert len(diagnostic.epoch_rows) == 4
            assert len(diagnostic.change_rows) == 1
            assert diagnostic.pending is None and diagnostic.val_parts == []
            assert diagnostic.handle is None and not runner.model._forward_hooks
            assert isinstance(diagnostic.previous, np.ndarray)
            manifest = json.loads((diagnostic.output_dir / "manifest.json").read_text())
            assert manifest["epochs"] == [0, 1]
            epoch = pd.read_csv(diagnostic.output_dir / "epoch_metrics.csv")
            assert list(epoch[epoch.split == "train"].n) == [27, 27]
        return runner, trace, rng_state

    baseline, a, rng_a = run(False)
    observed, b, rng_b = run(True)
    assert torch.equal(rng_a, rng_b)
    assert len(a.rows) == len(b.rows)
    for ar, br in zip(a.rows, b.rows):
        assert ar[0] == br[0]
        torch.testing.assert_close(ar[1], br[1], rtol=0, atol=0)
        for aa, bb in zip(ar[2] + ar[3] + ar[4], br[2] + br[3] + br[4]):
            torch.testing.assert_close(aa, bb, rtol=0, atol=0)
    # The observer's final extra validate happens after Trace's hook is removed.
    assert len(a.gates) == len(b.gates)
    for aa, bb in zip(a.gates, b.gates):
        torch.testing.assert_close(aa, bb, rtol=0, atol=0)
    for aa, bb in zip(baseline.parameters(), observed.parameters()):
        torch.testing.assert_close(aa, bb, rtol=0, atol=0)


def test_exception_cleanup(tmp_path):
    dataset = TinyDataset()
    callback = GatingDiagnostics(tmp_path, 1, {}, Subset(dataset, [0]), Subset(dataset, [1]))
    runner = NeuralSDERunner(**runner_config())
    callback.on_fit_start(None, runner)
    callback.on_exception(None, runner, RuntimeError("test"))
    assert not runner.model._forward_hooks and callback.handle is None


def test_bundled_checkpoint_still_loads():
    paths = list(Path("workspace/neural_SDE/logs/train_neural_SDE/US_Stocks/checkpoints").glob("*MoE.ckpt"))
    assert paths
    runner, checkpoint = load_moe_checkpoint(paths[0])
    x, f, *_ = window_features(np.linspace(10, 20, 252, dtype=np.float32)[None, :])
    with torch.inference_mode():
        p = runner(torch.from_numpy(x), torch.from_numpy(f))[2].numpy()
    validate_probabilities(p)


def test_offline_report_from_checkpoint_with_missing_events(tmp_path, monkeypatch):
    from experiments.neural_SDE.gating_diagnostics import analysis
    runner = CaptureRunner()
    runner.dset_cfg = {"lookback_window": 252}
    checkpoint = {"hyper_parameters": {"dset_cfg": runner.dset_cfg}, "epoch": 3}
    monkeypatch.setattr(analysis, "load_moe_checkpoint", lambda path: (runner, checkpoint))
    path = tmp_path / "model.ckpt"
    path.write_bytes(b"checkpoint loader is mocked for this report integration test")
    dates = pd.bdate_range("2017-01-01", "2018-06-01")
    prices = pd.DataFrame(dict(IssueId="00000001", Date=dates,
                               ClAdjLoc=100 + np.arange(len(dates)) * .01))
    source = tmp_path / "prices.txt"
    prices.to_csv(source, sep="\t", index=False)
    root = tmp_path / "analysis"
    report = analysis.analyze_checkpoint(path, root, issue_ids=["00000001"], data_path=source)
    assert len(list(root.glob("??_gating.png"))) == 8
    html = report.read_text(encoding="utf-8")
    assert html.count("data:image/png;base64,") == 8
    assert "cannot reconstruct" in html
    manifest = json.loads((root / "manifest.json").read_text())
    assert not manifest["training_history"] and manifest["evaluation_counts"]["events"] == 0
    samples = pd.read_csv(root / "real_gates.csv", dtype={"issue_id": str})
    arrays = np.load(root / "real_windows.npz")
    np.testing.assert_allclose(arrays["probabilities"], samples[list(PROBS)], atol=1e-7)
    bins = pd.read_csv(root / "distance_bins.csv")
    assert bins[bins.level == -1].n.sum() == samples["aggregate"].sum()
    assert not bins.supported.any()
