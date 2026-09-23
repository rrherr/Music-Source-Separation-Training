# Music Source Separation Inference

Inference-only fork of [ZFTurbo/Music-Source-Separation-Training](https://github.com/ZFTurbo/Music-Source-Separation-Training): separate music files into stems with pre-trained models. Training, validation and GUI code have been removed. Original repository is based on [kuielab code](https://github.com/kuielab/sdx23/tree/mdx_AB/my_submission/src) for [SDX23 challenge](https://github.com/kuielab/sdx23/tree/mdx_AB/my_submission/src). Brought to you by [MVSep.com](https://mvsep.com).

## Models

Model can be chosen with `--model_type` arg.

Available models:

* MDX23C based on [KUIELab TFC TDF v3 architecture](https://github.com/kuielab/sdx23/). Key: `mdx23c`.
* Demucs4HT [[Paper](https://arxiv.org/abs/2211.08553)]. Key: `htdemucs`.
* VitLarge23 based on [Segmentation Models Pytorch](https://github.com/qubvel/segmentation_models.pytorch). Key: `segm_models`.
* TorchSeg based on [TorchSeg module](https://github.com/qubvel/segmentation_models.pytorch). Key: `torchseg`.
* Band Split RoFormer [[Paper](https://arxiv.org/abs/2309.02612), [Repository](https://github.com/lucidrains/BS-RoFormer)] . Key: `bs_roformer`.
* Mel-Band RoFormer [[Paper](https://arxiv.org/abs/2310.01809), [Repository](https://github.com/lucidrains/BS-RoFormer)]. Key: `mel_band_roformer`.
* Swin Upernet [[Paper](https://arxiv.org/abs/2103.14030)] Key: `swin_upernet`.
* BandIt Plus [[Paper](https://arxiv.org/abs/2309.02539), [Repository](https://github.com/karnwatcharasupat/bandit)] Key: `bandit`.
* SCNet [[Paper](https://arxiv.org/abs/2401.13276), [Official Repository](https://github.com/starrytong/SCNet), [Unofficial Repository](https://github.com/amanteur/SCNet-PyTorch)] Key: `scnet`.
* BandIt v2 [[Paper](https://arxiv.org/abs/2407.07275), [Repository](https://github.com/kwatcharasupat/bandit-v2)] Key: `bandit_v2`.
* Apollo [[Paper](https://arxiv.org/html/2409.08514v1), [Repository](https://github.com/JusperLee/Apollo)] Key: `apollo`.
* BSMamba2 [[Paper](https://arxiv.org/abs/2508.14556), [Repository](https://github.com/EuiYeonKim/BSMamba2)] Key: `bs_mamba2`.
* Conformer [[Paper](https://arxiv.org/abs/2005.08100), [Repository](https://github.com/lucidrains/conformer)] Key: `conformer`.
* DTTNet [[Paper](https://arxiv.org/abs/2309.08684), [Repository](https://github.com/junyuchen-cjy/DTTNet-Pytorch)] Key: `dttnet`.
* BS Conformer Key: `bs_conformer`
* SCNet Tran Key: `scnet_tran`.
* SCNet Masked Key: `scnet_masked`.

1. **Note 1**: For `segm_models` there are many different encoders is possible. [Look here](https://github.com/qubvel/segmentation_models.pytorch#encoders-).
2. **Note 2**: Thanks to [@lucidrains](https://github.com/lucidrains) for recreating the RoFormer models based on papers.
3. **Note 3**: For `torchseg` gives access to more than 800 encoders from `timm` module. It's similar to `segm_models`.

## How to: Inference

### Inference example

```bash
python inference.py \
    --model_type mdx23c \
    --config_path configs/config_musdb18_mdx23c.yaml \
    --start_check_point results/last_mdx23c.ckpt \
    --input_folder input/wavs/ \
    --store_dir separation_results/
```

All inference parameters are in `parse_args_inference` in [utils/settings.py](utils/settings.py).
Convert models to ONNX and TensorRT formats [here](https://github.com/ZFTurbo/MSS_ONNX_TensorRT).

## Code description

* `configs/config_*.yaml` - configuration files for models
* `models/*` - set of available model architectures
* `inference.py` - process folder with music files and separate them
* `utils/` - model loading, demixing and audio helpers used by inference
* `ensemble.py` - useful script to ensemble results of different models to make results better (see [docs](docs/ensemble.md)).

## Pre-trained models

Look here: [List of Pre-trained models](docs/pretrained_models.md)

## Citation

* [arxiv paper](https://arxiv.org/abs/2305.07489)

```text
@misc{solovyev2023benchmarks,
      title={Benchmarks and leaderboards for sound demixing tasks}, 
      author={Roman Solovyev and Alexander Stempkovskiy and Tatiana Habruseva},
      year={2023},
      eprint={2305.07489},
      archivePrefix={arXiv},
      primaryClass={cs.SD}
}
```
