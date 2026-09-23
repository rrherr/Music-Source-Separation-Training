"""Chunked overlap-add inference: run a separation network over a whole recording.

Ported from ZFTurbo/Music-Source-Separation-Training, `utils/model_utils.py` (`demix` in its
generic mode and `_getWindowingArray`). See `demix_array` for how it differs from upstream.
"""
from __future__ import annotations

import contextlib
from dataclasses import dataclass
from typing import Callable, Optional, Tuple, Union

import numpy as np
import torch
from torch import nn


@dataclass(frozen=True)
class Chunk:
    """One model input window, in padded-signal coordinates."""

    start: int
    length: int
    first: bool
    last: bool

    @property
    def end(self) -> int:
        """One past the last sample."""
        return self.start + self.length


@dataclass(frozen=True)
class ChunkPlan:
    """Where the chunks fall for a signal of `length` samples."""

    length: int
    chunk_size: int
    step: int
    border: int
    fade_size: int
    padded_length: int
    chunks: Tuple[Chunk, ...]

    @property
    def padded(self) -> bool:
        """Whether the signal is reflect-padded by `border` on both sides."""
        return self.padded_length != self.length


def plan_chunks(length: int, chunk_size: int, num_overlap: int) -> ChunkPlan:
    """
    Decide padding and chunk positions for `demix_array`. Pure.

    Step is `chunk_size // num_overlap`; the signal is reflect-padded by `chunk_size - step`
    on each side when it is longer than twice that border; chunks start every `step` samples
    until the padded end, the final one being cut short. Every chunk knows whether it is the
    first or the last, so each gets its own window (see `window`).
    """
    if length <= 0 or chunk_size <= 0 or num_overlap <= 0:
        raise ValueError("length, chunk_size and num_overlap must be positive")
    step = max(1, chunk_size // num_overlap)
    border = chunk_size - step
    fade_size = chunk_size // 10 if num_overlap > 1 else 0
    padded = length + 2 * border if (length > 2 * border and border > 0) else length
    chunks = tuple(
        Chunk(start=s, length=min(chunk_size, padded - s), first=(s == 0), last=(s + step >= padded))
        for s in range(0, padded, step)
    )
    return ChunkPlan(length, chunk_size, step, border, fade_size, padded, chunks)


def window(chunk_size: int, fade_size: int, *, first: bool, last: bool) -> torch.Tensor:
    """Linear cross-fade window: ramps in unless `first`, ramps out unless `last`. Pure."""
    w = torch.ones(chunk_size)
    if fade_size > 0:
        if not first:
            w[:fade_size] = torch.linspace(0, 1, fade_size)
        if not last:
            w[-fade_size:] = torch.linspace(1, 0, fade_size)
    return w


def autocast_for(device: Union[str, torch.device], enabled: bool = True) -> contextlib.AbstractContextManager:
    """Mixed-precision context for `device`: float16 on cuda and mps, nothing on cpu. Pure."""
    device_type = torch.device(device).type
    if enabled and device_type in ("cuda", "mps"):
        return torch.autocast(device_type=device_type, dtype=torch.float16)
    return contextlib.nullcontext()


def demix_array(
    model: torch.nn.Module,
    mix: Union[np.ndarray, torch.Tensor],
    *,
    chunk_size: int,
    num_overlap: int,
    batch_size: int,
    device: Union[str, torch.device],
    use_amp: bool = True,
    on_batch: Optional[Callable[[int, int], None]] = None,
) -> np.ndarray:
    """
    Run `model` over `mix` in overlapping chunks and cross-fade the pieces back together.

    Ported from upstream MSST's `demix` (generic mode). Differences from the upstream loop:

    * Every chunk gets its own window from its own position. The old loop decided "first" and
      "last" once per batch, so with `batch_size > 1` the whole batch lost its fade; and it
      used `elif`, so a recording that fits in one chunk kept a fade-out.
    * With `num_overlap == 1` no fade is applied: there is no neighbour to cross-fade with,
      and the old loop zeroed the outermost samples and then divided by zero.
    * float16 autocast runs on MPS as well as CUDA when `use_amp` is set; CPU stays float32.
    * Progress is reported through `on_batch(samples_done, samples_total)` instead of tqdm.

    Args:
        model: maps (b, channels, chunk_size) to (b, instruments, channels, chunk_size), or
            to (b, channels, chunk_size) for a single instrument.
        mix: (channels, samples) audio.

    Returns:
        float32 array of shape (instruments, channels, samples), with exactly
        `mix.shape[-1]` samples.
    """
    x = torch.as_tensor(np.ascontiguousarray(mix, dtype=np.float32))
    if x.ndim != 2:
        raise ValueError("mix must be (channels, samples)")
    plan = plan_chunks(x.shape[-1], chunk_size, num_overlap)
    if plan.padded:
        x = nn.functional.pad(x, (plan.border, plan.border), mode="reflect")

    result: Optional[torch.Tensor] = None
    counter = torch.zeros(x.shape[-1])
    total = plan.padded_length

    with autocast_for(device, use_amp), torch.inference_mode():
        for batch_start in range(0, len(plan.chunks), batch_size):
            batch = plan.chunks[batch_start:batch_start + batch_size]
            parts = []
            for c in batch:
                part = x[:, c.start:c.end]
                pad = chunk_size - c.length
                if pad:
                    mode = "reflect" if c.length > chunk_size // 2 else "constant"
                    part = nn.functional.pad(part, (0, pad), mode=mode)
                parts.append(part)
            out = model(torch.stack(parts).to(device))
            if out.ndim == 3:
                out = out[:, None]
            out = out.float().cpu()
            if result is None:
                result = torch.zeros((out.shape[1],) + tuple(x.shape))
            for j, c in enumerate(batch):
                w = window(chunk_size, plan.fade_size, first=c.first, last=c.last)[:c.length]
                result[..., c.start:c.end] += out[j, ..., :c.length] * w
                counter[c.start:c.end] += w
            if on_batch is not None:
                on_batch(min(batch[-1].end, total), total)

    assert result is not None  # plan_chunks always yields at least one chunk
    estimate = torch.nan_to_num(result / counter, nan=0.0)
    if plan.padded:
        estimate = estimate[..., plan.border:-plan.border]
    return np.asarray(estimate.numpy(), dtype=np.float32)
