# msst: BS-RoFormer inference

A library-only, BS-RoFormer-only fork of
[ZFTurbo/Music-Source-Separation-Training](https://github.com/ZFTurbo/Music-Source-Separation-Training)
(MIT). It keeps the Band-Split RoFormer network
([paper](https://arxiv.org/abs/2309.02612), recreated by [@lucidrains](https://github.com/lucidrains/BS-RoFormer))
and the chunked overlap-add loop that runs it over a whole recording. Training, validation,
the GUI, the command line, configs and every other architecture have been removed.

## Install

```bash
pip install "music-source-separation-training @ git+https://github.com/rrherr/Music-Source-Separation-Training"
```

Dependencies: torch, numpy, einops, rotary-embedding-torch.

## Use

```python
import torch
import msst

model = msst.BSRoformer(**model_kwargs)          # the `model:` section of an MSST YAML config
model.load_state_dict(torch.load("model.ckpt", map_location="cpu", weights_only=True))
model = model.to(device).eval()

stems = msst.demix_array(                        # mix: (channels, samples) float32
    model, mix,
    chunk_size=441000, num_overlap=2, batch_size=1, device=device,
    on_batch=lambda done, total: ...,            # optional progress callback
)                                                 # -> (instruments, channels, samples) float32
```

Set `use_torch_checkpoint` to false in `model_kwargs`; it only saves memory during training.
`use_pope` is not supported.

`demix_array` differs from upstream's `demix` on purpose: every chunk gets its own fade
window (upstream decides first/last once per batch, so `batch_size > 1` loses fades), no fade
is applied when `num_overlap == 1`, and float16 autocast runs on MPS as well as CUDA
(`use_amp=False` keeps float32). `plan_chunks` and `window` are the pure pieces it is built
from.

## Changes to the network

`msst/bs_roformer.py` and `msst/attend.py` come from upstream `models/bs_roformer/` with
these edits only: `beartype` and `packaging` are no longer used, the optional PoPE
positional embedding is removed, and attention uses `torch.nn.attention.sdpa_kernel`
instead of the deprecated `sdp_kernel`. Checkpoints load strictly and produce identical
output.

## Tests

```bash
pip install -e . --group test
pytest
```
