"""Temporal metric edge cases and isolated integration with the existing rollout."""

import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from experiments.neural_SDE.gating_diagnostics.temporal_statistics import (
    trajectory_metrics, states_and_flags, invalid_reason, paired_weights,
    compare_trajectories, dwell_tables, distribution_tables,
)
from experiments.neural_SDE.gating_diagnostics.temporal import origin_seed


def identity(sid=0, source="real", path=-1, stock="00000001"):
    return dict(segment_id=sid, issue_id=stock, origin_date="2018-01-02", source=source, path_id=path)


def test_constant_and_alternating_states():
    for states, switches in [(np.full(101, 2), 0), (np.arange(101) % 2, 100)]:
        p = np.eye(3)[states]
        row, dwell, changes = trajectory_metrics(p, identity())
        assert row["switch_count"] == row["switches_per_100"] == switches
        assert sum(r["duration"] for r in dwell) == 101
        assert sum(row[f"count_{i}_{j}"] for i in range(3) for j in range(3)) == 100
        assert dwell[0]["left_censored"] and dwell[-1]["right_censored"]
        assert len(changes) == 100
    row, dwell, _ = trajectory_metrics(np.tile([.03, .32, .65], (101, 1)), identity())
    assert len(dwell) == 1 and dwell[0]["duration"] == 101 and not dwell[0]["interior"]
    assert row["transition_2_2"] == 1 and np.isnan(row["transition_0_0"])


def test_dwell_example_and_soft_changes_with_ties():
    states = [2, 2, 2, 0, 0, 2]
    _, runs, _ = trajectory_metrics(np.eye(3)[states], identity())
    assert [r["duration"] for r in runs] == [3, 2, 1]
    assert [r["interior"] for r in runs] == [False, True, False]
    p = np.array([[.03, .48, .49], [.03, .49, .48], [1/3, 1/3, 1/3]])
    state, margin, ambiguous, ties = states_and_flags(p)
    np.testing.assert_array_equal(state, [2, 1, 0])
    assert ambiguous.all() and ties[-1]
    row, _, changes = trajectory_metrics(p, identity())
    assert row["switch_count"] == 2 and changes[0] == pytest.approx(.02/3)


def test_invalid_paths_and_seed_derivation():
    p, f = np.tile([.1, .2, .7], (101, 1)), np.zeros((101, 7))
    assert invalid_reason(np.ones(101), p, f) == ""
    assert invalid_reason(np.zeros(101), p, f) == "nonpositive price"
    assert invalid_reason(np.full(101, np.nan), p, f) == "nonfinite price"
    assert invalid_reason(np.ones(101), p * 2, f) == "invalid gate probabilities"
    assert origin_seed(1, "a", "2018-01-02") == origin_seed(1, "a", "2018-01-02")
    assert origin_seed(1, "a", "2018-01-02") != origin_seed(1, "b", "2018-01-02")


def test_equal_origin_weights_paired_bootstrap_and_absent_states():
    rows, dwells, soft = [], [], {}
    for sid in range(6):
        # Different ensemble sizes must not give different origin weights.
        for source, paths in [("real", [-1]), ("synthetic", list(range(sid + 1)))]:
            for path in paths:
                p = np.tile([.01 + sid * .01, .2, .79 - sid * .01], (101, 1))
                ident = identity(sid, source, path, str(sid))
                row, dd, delta = trajectory_metrics(p, ident)
                rows.append(dict(row, valid=True))
                dwells.extend(dd)
                soft[(sid, source, path)] = delta
    rows += [dict(identity(99), valid=True), dict(identity(99, "synthetic", 0), valid=False)]
    t = paired_weights(pd.DataFrame(rows))
    np.testing.assert_allclose(t[t.included].groupby(["segment_id", "source"]).weight.sum(), 1)
    assert not t[t.segment_id == 99].included.any()
    before = np.random.get_state()
    summary, transition, counts = compare_trajectories(t, seed=1, bootstrap=100)
    np.testing.assert_array_equal(np.random.get_state()[1], before[1])
    difference = summary[summary.source == "synthetic-minus-real"]
    np.testing.assert_allclose(difference[["mean", "low", "high"]], 0, atol=1e-14)
    assert transition[transition.from_expert == "Bounce"].probability.isna().all()
    hover = transition[(transition.source == "real") & (transition.from_expert == "Hover")]
    assert hover.supported.all() and hover.raw_source_count.iloc[0] == 600
    assert counts.groupby("source").weighted_count.sum().tolist() == [600, 600]
    _, distribution, ds = dwell_tables(pd.DataFrame(dwells), t)
    assert ds[(ds.expert == "Hover") & (ds.kind == "all observed")].censored_fraction.eq(1).all()
    bins = distribution_tables(t, soft)
    np.testing.assert_allclose(bins.groupby(["metric", "source"]).probability.sum(), 1)
    pd.testing.assert_frame_equal(summary, compare_trajectories(t, seed=1, bootstrap=100)[0])


def test_no_valid_pairs():
    t = paired_weights(pd.DataFrame([dict(identity(), valid=False), dict(identity(source="synthetic", path=0), valid=False)]))
    assert not t.included.any() and t.weight.sum() == 0
    assert all(d.empty for d in compare_trajectories(t))


def test_existing_rollout_parity_causality_and_full_report(tmp_path):
    # generate_samples configures inter-op threads at import time: exercise it
    # in its own process, as the production temporal CLI does.
    script = r'''
import hashlib, json, sys
from pathlib import Path
from experiments.neural_SDE.generate_samples import _rollout_sde
import numpy as np
import pandas as pd
import torch
from experiments.neural_SDE.trainer_cfg.neural_SDE.US_Stocks import get_trainer_cfg
from experiments.neural_SDE.gating_diagnostics.analysis import load_moe_checkpoint
from experiments.neural_SDE.gating_diagnostics.temporal import paired_origin, run_temporal, evaluate_windows
from experiments.neural_SDE.gating_diagnostics.statistics import window_features, PROBS
from amgm.models.neural_SDE.runner import NeuralSDERunner
root = Path(sys.argv[1]); source = root/'baseline'; source.mkdir()
torch.manual_seed(1)
cfg = get_trainer_cfg(); cfg['compile_model'] = False
runner = NeuralSDERunner(**cfg).eval()
checkpoint = source/'model.ckpt'
torch.save(dict(hyper_parameters=cfg, state_dict=runner.state_dict(), epoch=8), checkpoint)
dates = pd.bdate_range('2017-01-02', periods=370)
values = (100 + .015*np.arange(len(dates)) + np.sin(np.arange(len(dates)) / 8)).astype(np.float32)
origin = 260; history = values[origin-251:origin+1]; future = values[origin:origin+101]
prices, gates, features = paired_origin(runner, _rollout_sde, history, future, 3, 9, 1.)
x, *_ = window_features(history[None])
batch = dict(price_window=torch.from_numpy(np.repeat(x,3,axis=0)), sample_min=torch.full((3,1),float(history.min())), sample_range=torch.full((3,1),float(np.ptp(history))))
px, pi, _, _ = _rollout_sde(runner,batch,100,1.,9)
np.testing.assert_array_equal(prices[1:], px)
np.testing.assert_array_equal(gates[1:,:100], pi)
reconstructed = (batch['price_window']*batch['sample_range']+batch['sample_min']).numpy()
terminal = np.concatenate([reconstructed,px[:,1:]],axis=1)[:,-252:].copy()
last_pi,last_f = evaluate_windows(runner,terminal)
np.testing.assert_array_equal(gates[1:,-1],last_pi)
np.testing.assert_array_equal(features[1:,-1],last_f)
changed = future.copy(); changed[50:] += 5
px2,pi2,f2 = paired_origin(runner,_rollout_sde,history,changed,3,9,1.)
np.testing.assert_array_equal(gates[0,:50],pi2[0,:50])
np.testing.assert_array_equal(prices[1:],px2[1:])
data = root/'prices.txt'
pd.DataFrame(dict(IssueId='00000001',Date=dates,ClAdjLoc=values)).to_csv(data,sep=chr(9),index=False)
date = str(dates[origin])[:10]
pd.DataFrame([dict(segment_id=0,issue_id='00000001',origin_date=date,horizon=100)]).to_csv(source/'segments.csv',index=False)
ref = pd.DataFrame(dict(segment_id=0,issue_id='00000001',step=np.arange(101),price=future))
for k,p in enumerate(PROBS): ref[p]=gates[0,:,k]
ref.to_csv(source/'real_gates.csv',index=False)
(source/'manifest.json').write_text(json.dumps(dict(checkpoint=str(checkpoint),checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),checkpoint_epoch=8,data_source=str(data))))
report = run_temporal(source,root/'output',_rollout_sde,paths_per_origin=3)
assert report.read_text().count('data:image/png;base64,') == 5
assert len(list(report.parent.glob('??_temporal.png'))) == 5
t=pd.read_csv(report.parent/'trajectories.csv')
assert len(t)==4 and t[t.valid].observations.eq(101).all() and t[t.valid].transitions.eq(100).all()
steps=pd.read_csv(report.parent/'steps.csv.gz')
assert len(steps)==404
assert json.loads((report.parent/'manifest.json').read_text())['checkpoint_unchanged']
from experiments.neural_SDE.gating_diagnostics.temporal_plots import render_temporal_report
render_temporal_report(report.parent)
'''
    result = subprocess.run([sys.executable, "-c", script, str(tmp_path)], capture_output=True, text=True,
                            env={**os.environ, "MPLBACKEND": "Agg"}, timeout=180)
    assert result.returncode == 0, result.stdout + result.stderr
