import numpy as np
import pytest
import torch

from amgm.models.neural_SDE.gate_features import (
    FIB_RATIOS, GATE_FEATURE_SETS, PRICE_V1_NAMES, compute_gate_features, gate_feature_names,
)

W = 252
COL = {name: i for i, name in enumerate(PRICE_V1_NAMES)}


def gbm(batch=64, seed=0, vol=0.02):
    rng = np.random.default_rng(seed)
    log_p = np.log(50.0) + np.cumsum(rng.normal(0, vol, size=(batch, W)), axis=1)
    return np.exp(log_p)


def normalise(raw):
    lo, hi = raw.min(1, keepdims=True), raw.max(1, keepdims=True)
    return torch.from_numpy(((raw - lo) / (hi - lo)).astype(np.float32))


@pytest.mark.parametrize("name", sorted(GATE_FEATURE_SETS))
def test_shapes_names_and_finite(name):
    g = compute_gate_features(name, normalise(gbm()))
    assert g.shape == (64, len(gate_feature_names(name)))
    assert torch.isfinite(g).all()


def test_unknown_set_and_short_window_raise():
    with pytest.raises(ValueError, match="gate_feature_set"):
        compute_gate_features("nope", torch.zeros(2, W))
    with pytest.raises(ValueError, match="W>=61"):
        compute_gate_features("price_v1", torch.rand(2, 60))


def test_level_distance_matches_raw_price_definition():
    raw = gbm(batch=400, seed=1)
    g = compute_gate_features("price_v1", normalise(raw)).numpy()
    lo, hi = raw.min(1, keepdims=True), raw.max(1, keepdims=True)
    levels = lo + np.asarray(FIB_RATIOS) * (hi - lo)
    p_t = raw[:, -1:]
    sigma20 = np.diff(np.log(raw), axis=1)[:, -20:].std(1, keepdims=True)
    reference = (p_t - levels) / (sigma20 * p_t)
    ours = g[:, :7]
    keep = np.abs(reference) < 4.0  # away from the clip
    assert np.corrcoef(ours[keep], reference[keep])[0, 1] > 0.99
    assert np.median(np.abs(ours[keep] - reference[keep]) / np.abs(reference[keep]).clip(0.1)) < 0.05


def test_causal_and_invariant_to_old_history():
    x = normalise(gbm(batch=8, seed=2))
    changed = x.clone()
    changed[:, 10:150] = changed[:, 10:150].flip(1)  # keeps window min/max, alters old history
    if not (torch.equal(changed.min(1).values, x.min(1).values)
            and torch.equal(changed.max(1).values, x.max(1).values)):
        pytest.skip("min/max moved")
    torch.testing.assert_close(compute_gate_features("price_v1", changed),
                               compute_gate_features("price_v1", x))


def window_ending(tail, base=0.3):
    """Normalised window: low 0 and high 1 early, then a noisy middle and a chosen tail."""
    x = np.full(W, base, dtype=np.float32)
    x[0], x[1] = 0.0, 1.0
    x[2:W - len(tail)] = base + 0.01 * np.sin(np.arange(W - len(tail) - 2))
    x[W - len(tail):] = tail
    return torch.from_numpy(x[None, :])


def test_approach_sign():
    # Nearest level 0.5. Moving up from below means approaching; moving down from below means leaving.
    up = compute_gate_features("price_v1", window_ending([0.40, 0.45]))
    down = compute_gate_features("price_v1", window_ending([0.47, 0.45]))
    assert up[0, COL["approach"]] == 1.0
    assert down[0, COL["approach"]] == -1.0


def test_zone_days_streak_cap_and_reset():
    in_zone = [0.5, 0.505, 0.495]
    streak = compute_gate_features("price_v1", window_ending([0.3] + in_zone))
    assert streak[0, COL["zone_days"]] == pytest.approx(3 / 20)
    long = compute_gate_features("price_v1", window_ending([0.5] * 30))
    assert long[0, COL["zone_days"]] == pytest.approx(1.0)
    reset = compute_gate_features("price_v1", window_ending([0.5] * 5 + [0.6, 0.5]))
    assert reset[0, COL["zone_days"]] == pytest.approx(1 / 20)


def test_clip_and_constant_tail_eps():
    flat = window_ending([0.3] * 80)
    g = compute_gate_features("price_v1", flat, clip=5.0)
    assert torch.isfinite(g).all()
    assert g.abs().max() <= 5.0
    assert g[0, COL["dist_vol_0"]] == 5.0  # far above level 0 in tiny-vol units


# --- NeuralSDEMoE gate_inputs --------------------------------------------------------------

from amgm.models.mlp import NeuralSDEMoE  # noqa: E402


def moe(**kwargs):
    torch.manual_seed(0)
    return NeuralSDEMoE(lookback_window=W, num_features=7, hidden_sizes=[32, 16], **kwargs)


def model_inputs(batch=16):
    x = normalise(gbm(batch=batch, seed=3))
    f = x[:, -1:] - torch.tensor(FIB_RATIOS)
    return x, f


def test_default_gate_is_the_legacy_architecture():
    legacy, explicit = moe(), moe(gate_inputs="h")
    assert legacy.gate_feature_names == ()
    assert {k: v.shape for k, v in legacy.state_dict().items()} == \
        {k: v.shape for k, v in explicit.state_dict().items()}
    assert legacy.gate_head.fc_gate.in_features == 16 + 7
    x, f = model_inputs()
    for a, b in zip(legacy(x, f), explicit(x, f)):
        torch.testing.assert_close(a, b)


@pytest.mark.parametrize("mode,set_name,in_dim", [
    ("features", "price_v1", 14), ("features", "vol_only", 2), ("h+features", "price_v1", 16 + 7 + 14)])
def test_feature_gates_forward_and_train(mode, set_name, in_dim):
    model = moe(gate_inputs=mode, gate_feature_set=set_name)
    assert model.gate_head.fc_gate.in_features == in_dim
    x, f = model_inputs()
    mu, sigma, pi = model(x, f)
    assert mu.shape == sigma.shape == (16, 1)
    torch.testing.assert_close(pi.sum(1), torch.ones(16))
    (mu.sum() + sigma.sum()).backward()
    assert model.gate_head.fc_gate.weight.grad.abs().sum() > 0
    assert model.gate_features(x).shape == (16, len(model.gate_feature_names))


def test_feature_gate_ignores_encoder_path_but_uses_gate_features():
    model = moe(gate_inputs="features")
    x, f = model_inputs()
    _, _, pi = model(x, f)
    _, _, pi_other_f = model(x, f + 0.3)  # f feeds experts only in this mode
    torch.testing.assert_close(pi, pi_other_f)


def test_invalid_gate_configuration_raises():
    with pytest.raises(ValueError, match="gate_inputs"):
        moe(gate_inputs="bogus")
    with pytest.raises(ValueError, match="gate_feature_set"):
        moe(gate_inputs="features", gate_feature_set="bogus")
    with pytest.raises(ValueError, match="lookback_window"):
        NeuralSDEMoE(lookback_window=12, gate_inputs="features")
    NeuralSDEMoE(lookback_window=12)  # legacy gate keeps working with short test windows


def test_exp2_exports_gate_columns_only_for_long_windows():
    from experiments.neural_SDE.gating_diagnostics.analysis import export_gate_features
    columns = export_gate_features(normalise(gbm(batch=4)).numpy())
    assert list(columns) == [f"gate_{name}" for name in PRICE_V1_NAMES]
    assert all(v.shape == (4,) for v in columns.values())
    assert export_gate_features(np.random.rand(4, 12).astype(np.float32)) == {}
