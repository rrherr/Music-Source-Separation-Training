"""Smoke test: run inference.py end-to-end on CPU with a tiny randomly initialized MDX23C model."""
import os
import sys

import numpy as np
import soundfile as sf
import torch
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from inference import proc_folder
from utils.settings import get_model_from_config


def make_tiny_config(tmp_path):
    with open(os.path.join(ROOT, 'configs', 'config_vocals_mdx23c.yaml')) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    config['model'].update(num_channels=8, growth=8, num_blocks_per_scale=1, num_scales=2)
    config['inference'] = {'batch_size': 1, 'num_overlap': 1, 'dim_t': 256}
    config_path = tmp_path / 'config.yaml'
    with open(config_path, 'w') as f:
        yaml.dump(config, f)
    return str(config_path), config


def test_inference_mdx23c(tmp_path):
    config_path, config = make_tiny_config(tmp_path)
    model, _ = get_model_from_config('mdx23c', config_path)
    checkpoint_path = tmp_path / 'model.ckpt'
    torch.save(model.state_dict(), checkpoint_path)

    input_dir = tmp_path / 'input'
    input_dir.mkdir()
    sr = config['audio']['sample_rate']
    audio = np.random.default_rng(0).uniform(-0.5, 0.5, size=(3 * sr, 2)).astype(np.float32)
    sf.write(input_dir / 'song.wav', audio, sr)

    store_dir = tmp_path / 'output'
    proc_folder({
        'model_type': 'mdx23c',
        'config_path': config_path,
        'start_check_point': str(checkpoint_path),
        'input_folder': str(input_dir),
        'store_dir': str(store_dir),
        'force_cpu': True,
        'disable_detailed_pbar': True,
    })

    for instr in config['training']['instruments']:
        out, out_sr = sf.read(store_dir / 'song' / f'{instr}.wav')
        assert out_sr == sr
        assert out.shape == audio.shape
        assert np.isfinite(out).all()
