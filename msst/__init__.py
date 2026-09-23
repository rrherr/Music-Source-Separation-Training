"""Library entry points for this fork, under one package name.

The upstream layout installs `models`, `utils`, `configs`, `inference` and `ensemble` as
top-level names. They stay where they are so merges from upstream stay clean; this package
re-exports what other programs need, so they can `import msst` instead of `import utils`.

Importing it needs only the core dependencies plus the `bs_roformer` extra: none of the
CLI extra (librosa, matplotlib, pandas, ml-collections, omegaconf, soundfile) is imported.
"""
from models.bs_roformer.bs_roformer import BSRoformer
from utils.model_utils import Chunk, ChunkPlan, autocast_for, demix_array, plan_chunks, window

__all__ = [
    "BSRoformer",
    "Chunk",
    "ChunkPlan",
    "autocast_for",
    "demix_array",
    "plan_chunks",
    "window",
]
