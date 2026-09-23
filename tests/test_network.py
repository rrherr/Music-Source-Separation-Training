"""BSRoformer builds, runs on CPU, and round-trips its weights strictly."""
import os
import sys

import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from msst import BSRoformer

TINY = dict(
    dim=32,
    depth=1,
    stereo=True,
    num_stems=1,
    time_transformer_depth=1,
    freq_transformer_depth=1,
    freqs_per_bands=(256, 257),
    dim_freqs_in=513,
    dim_head=16,
    heads=2,
    stft_n_fft=1024,
    stft_hop_length=256,
    stft_win_length=1024,
    mask_estimator_depth=1,
    mlp_expansion_factor=2,
)


@pytest.mark.parametrize("num_stems", [1, 2])
def test_forward_shape(num_stems):
    model = BSRoformer(**{**TINY, 'num_stems': num_stems}).eval()
    n = 8 * TINY['stft_hop_length']
    with torch.inference_mode():
        out = model(torch.randn(2, 2, n))
    assert out.shape == (2, num_stems, 2, n)
    assert torch.isfinite(out).all()


def test_state_dict_round_trips_strictly():
    torch.manual_seed(0)
    a = BSRoformer(**TINY).eval()
    b = BSRoformer(**TINY).eval()
    b.load_state_dict(a.state_dict(), strict=True)
    x = torch.randn(1, 2, 4 * TINY['stft_hop_length'])
    with torch.inference_mode():
        assert torch.equal(a(x), b(x))


def test_use_pope_is_rejected():
    with pytest.raises(ValueError):
        BSRoformer(**TINY, use_pope=True)
