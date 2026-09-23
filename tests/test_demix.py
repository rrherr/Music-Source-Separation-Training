"""demix: chunk planning, windows and overlap-add reconstruction.

Moved here from bluegrass-karaoke along with the code it tests. A gain model makes the
expected output exact: any window that does not sum to one shows up as an error.
"""
import os
import subprocess
import sys
import warnings

import numpy as np
import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import msst
from utils.model_utils import autocast_for, demix

lengths = st.integers(min_value=1, max_value=1200)
chunk_sizes = st.integers(min_value=4, max_value=200)
overlaps = st.integers(min_value=1, max_value=6)
batch_sizes = st.integers(min_value=1, max_value=4)


class GainModel(torch.nn.Module):
    """Returns `gains[k] * input` for each instrument k, as (b, instruments, c, t).

    Records the largest batch it saw so tests can check batching.
    """

    def __init__(self, *gains: float) -> None:
        super().__init__()
        self.gains = gains or (1.0,)
        self.max_batch = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.max_batch = max(self.max_batch, x.shape[0])
        return torch.stack([x * g for g in self.gains], dim=1)


def _run(model, mix, chunk_size, num_overlap, batch_size, on_batch=None):
    return msst.demix_array(
        model,
        mix,
        chunk_size=chunk_size,
        num_overlap=num_overlap,
        batch_size=batch_size,
        device="cpu",
        on_batch=on_batch,
    )


# --- plan_chunks --------------------------------------------------------------------------------


@given(lengths, chunk_sizes, overlaps)
def test_plan_covers_every_padded_sample_in_order(length, chunk_size, num_overlap):
    plan = msst.plan_chunks(length, chunk_size, num_overlap)
    assert plan.chunks[0].start == 0
    assert plan.chunks[0].first
    assert plan.chunks[-1].last
    assert plan.chunks[-1].end == plan.padded_length
    starts = [c.start for c in plan.chunks]
    assert starts == list(range(0, plan.padded_length, plan.step))
    assert all(0 < c.length <= chunk_size for c in plan.chunks)
    covered = np.zeros(plan.padded_length, dtype=int)
    for c in plan.chunks:
        covered[c.start:c.end] += 1
    assert (covered >= 1).all()
    assert sum(c.first for c in plan.chunks) == 1
    assert sum(c.last for c in plan.chunks) == 1


@given(lengths, chunk_sizes, overlaps)
def test_plan_padding_rule_matches_the_original_loop(length, chunk_size, num_overlap):
    plan = msst.plan_chunks(length, chunk_size, num_overlap)
    step = max(1, chunk_size // num_overlap)
    border = chunk_size - step
    if length > 2 * border and border > 0:
        assert plan.padded
        assert plan.padded_length == length + 2 * border
    else:
        assert not plan.padded
        assert plan.padded_length == length
    assert plan.fade_size == (chunk_size // 10 if num_overlap > 1 else 0)


def test_plan_rejects_nonpositive():
    with pytest.raises(ValueError):
        msst.plan_chunks(0, 10, 2)
    with pytest.raises(ValueError):
        msst.plan_chunks(10, 0, 2)


# --- window -------------------------------------------------------------------------------------


@given(chunk_sizes, st.booleans(), st.booleans())
def test_window_fades_only_toward_neighbours(chunk_size, first, last):
    fade = chunk_size // 10
    w = msst.window(chunk_size, fade, first=first, last=last)
    assert w.shape == (chunk_size,)
    assert ((w >= 0) & (w <= 1)).all()
    if fade > 1:  # a one-sample linspace is degenerate
        assert (w[0] == 1) == first
        assert (w[-1] == 1) == last
        assert (w[fade:chunk_size - fade] == 1).all()
    elif fade == 0:
        assert (w == 1).all()


def test_window_without_fade_is_all_ones():
    assert (msst.window(50, 0, first=False, last=False) == 1).all()


# --- demix_array --------------------------------------------------------------------------------


@settings(max_examples=200, deadline=None)
@given(lengths, chunk_sizes, overlaps, batch_sizes, st.floats(0.1, 2.0))
def test_demix_of_a_gain_model_is_the_gain(length, chunk_size, num_overlap, batch_size, gain):
    rng = np.random.default_rng(length * 7919 + chunk_size)
    mix = rng.standard_normal((2, length)).astype(np.float32)
    out = _run(GainModel(gain), mix, chunk_size, num_overlap, batch_size)
    assert out.shape == (1,) + mix.shape
    assert out.dtype == np.float32
    np.testing.assert_allclose(out[0], gain * mix, rtol=1e-4, atol=1e-5)


def test_demix_keeps_every_instrument():
    mix = np.random.default_rng(1).standard_normal((2, 700)).astype(np.float32)
    out = _run(GainModel(0.5, 2.0, -1.0), mix, 64, 4, 3)
    assert out.shape == (3, 2, 700)
    np.testing.assert_allclose(out, np.stack([0.5 * mix, 2.0 * mix, -mix]), rtol=1e-4, atol=1e-5)


@given(lengths, chunk_sizes, overlaps, batch_sizes)
def test_demix_reports_monotone_progress_ending_at_total(length, chunk_size, num_overlap, batch_size):
    seen = []
    _run(
        GainModel(),
        np.zeros((2, length), np.float32),
        chunk_size,
        num_overlap,
        batch_size,
        lambda d, t: seen.append((d, t)),
    )
    plan = msst.plan_chunks(length, chunk_size, num_overlap)
    dones = [d for d, _ in seen]
    assert dones == sorted(dones)
    assert dones[-1] == plan.padded_length
    assert {t for _, t in seen} == {plan.padded_length}
    assert len(seen) == -(-len(plan.chunks) // batch_size)


def test_demix_batches_chunks():
    model = GainModel()
    _run(model, np.zeros((2, 1000), np.float32), 100, 2, 3)
    assert model.max_batch == 3


def test_demix_accepts_three_dim_model_output():
    class Flat(torch.nn.Module):
        def forward(self, x):
            return x

    mix = np.random.default_rng(0).standard_normal((2, 300)).astype(np.float32)
    np.testing.assert_allclose(_run(Flat(), mix, 64, 2, 1)[0], mix, atol=1e-5)


def test_demix_rejects_wrong_rank():
    with pytest.raises(ValueError):
        _run(GainModel(), np.zeros(10, np.float32), 4, 2, 1)


@pytest.mark.parametrize("device, expect_autocast", [("cuda", True), ("mps", True), ("cpu", False)])
def test_autocast_is_fp16_on_accelerators_only(device, expect_autocast):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # torch warns when cuda is absent on this machine
        ctx = autocast_for(device)
    if expect_autocast:
        assert isinstance(ctx, torch.autocast)
        assert ctx.device == device
        assert ctx.fast_dtype == torch.float16
    else:
        assert not isinstance(ctx, torch.autocast)


def test_autocast_can_be_disabled():
    assert not isinstance(autocast_for("mps", enabled=False), torch.autocast)


# --- demix (ConfigDict wrapper) -----------------------------------------------------------------


def _config(**inference):
    from ml_collections import ConfigDict

    return ConfigDict({
        'audio': {'chunk_size': 64},
        'inference': {'num_overlap': 2, 'batch_size': 2, **inference},
        'training': {'instruments': ['vocals', 'other'], 'target_instrument': None, 'use_amp': False},
    })


@pytest.mark.parametrize("pbar", [False, True])
def test_wrapper_names_the_instruments_and_matches_the_core(pbar):
    mix = np.random.default_rng(2).standard_normal((2, 500)).astype(np.float32)
    out = demix(_config(), GainModel(1.0, 0.25), mix, torch.device('cpu'), 'bs_roformer', pbar=pbar)
    assert list(out) == ['vocals', 'other']
    core = _run(GainModel(1.0, 0.25), mix, 64, 2, 2)
    np.testing.assert_array_equal(out['vocals'], core[0])
    np.testing.assert_array_equal(out['other'], core[1])


def test_wrapper_prefers_inference_chunk_size():
    mix = np.random.default_rng(3).standard_normal((2, 500)).astype(np.float32)
    out = demix(_config(chunk_size=100), GainModel(1.0, 1.0), mix, 'cpu', 'bs_roformer')
    np.testing.assert_array_equal(out['vocals'], _run(GainModel(1.0), mix, 100, 2, 2)[0])


# --- the msst package ---------------------------------------------------------------------------


def test_importing_msst_does_not_pull_in_cli_dependencies():
    heavy = ['librosa', 'matplotlib', 'pandas', 'ml_collections', 'omegaconf', 'soundfile']
    code = f"import sys, msst; print([m for m in {heavy!r} if m in sys.modules])"
    out = subprocess.run([sys.executable, '-c', code], cwd=ROOT, capture_output=True, text=True, check=True)
    assert out.stdout.strip() == '[]'
