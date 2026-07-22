# MonoUNet

This repo contains the training, evaluation, and analysis code for MonoUNet variants.

## Environment Setup

Set up the environment using Pip as follows:

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Data Setup

The training and validation code expects nnU-Net style paths through environment variables:

```bash
export nnUNet_raw=data/nnUNet_raw
export nnUNet_preprocessed=data/nnUNet_preprocessed
export NO_ALBUMENTATIONS_UPDATE=1
```

For a dataset such as `Dataset073_GE_LE`, the expected structure is:

```text
data/
  nnUNet_raw/
    Dataset073_GE_LE/
      dataset.json
      imagesTr/
      labelsTr/
      imagesTs/
      labelsTs/
  nnUNet_preprocessed/
    Dataset073_GE_LE/
      splits_final.json
```

`splits_final.json` provides the train/validation folds used by `train.py`.

## How To Run MonoUNet

Most users should run MonoUNet through `[run_train.sh](/home/ultrai/UltrAi/monounet/run_train.sh)`. It is the recommended entrypoint for training, evaluation, and analysis.

Edit that script first to choose:

- `dataset_name`
- `all_archs`
- `train`, `eval`, and `analyze`
- `epochs`
- `b`
- `gpu`

Then run:

```bash
bash run_train.sh
```

At the moment, the script is set up to train `MonoUNetE123V2Gated` on `Dataset073_GE_LE`, loop over folds `0..2`, and then optionally run evaluation and model analysis.

Data augmentation is enabled by default in both `run_train.sh` and the Python CLI entrypoints.

If you want to run training directly instead of using the shell script:

```bash
python train.py \
  --dataset Dataset073_GE_LE \
  --arch MonoUNetE123V2Gated \
  --lr 0.01 \
  --epochs 1000 \
  --input_w 256 \
  --input_h 256 \
  --b 8 \
  --fold 0 \
  --min_lr 1e-5 \
  --loss TinyUNetLoss \
  --optimizer AdamW \
  --scheduler PolyLR \
  --weight_decay 0.01 \
  --input_channels 1 \
  --deep_supervision False \
  --data_augmentation True \
  --num_classes 1
```

## Evaluation

To evaluate a trained checkpoint manually:

```bash
python val.py \
  --name MonoUNetE123V2Gated \
  --train_dataset Dataset073_GE_LE \
  --train_fold 0 \
  --test_dataset Dataset070_Clarius_L15 \
  --test_split Tr \
  --save_preds True \
  --ckpt model_best.pth \
  --deep_supervision False \
  --data_augmentation True \
  --overlay False \
  --largest_component True
```

Use `test_split Ts` when evaluating on held-out test data stored under `imagesTs` and `labelsTs`.

## Where Results Go

Training outputs are saved under:

```text
models/<arch_name>/<dataset>/fold_<fold>/
```

Example:

```text
models/MonoUNetE123V2Gated/Dataset073_GE_LE/fold_0/
```

Important files in that folder:

- `config.yml`: run configuration
- `log.csv`: per-epoch train and validation metrics
- `loss_curves.png`: plotted loss and dice curves
- `model_best.pth`: best checkpoint by validation dice
- `model_final.pth`: final checkpoint at the end of training
- `model_latest.pth`: resume checkpoint written periodically

Mono-specific parameter logs are saved under:

```text
models/<arch_name>/<dataset>/fold_<fold>/mono_params/
```

Evaluation artifacts are saved under:

```text
models/<arch_name>/<train_dataset>/fold_<fold>/test/<test_dataset>/
```

Depending on flags, that directory can contain:

- `preds/`
- `overlays/`

If analysis is enabled in `run_train.sh`, model complexity metrics are saved to:

```text
models/<arch_name>/<dataset>/model_analysis.json
```

## Acknowledgements

This codebase uses certain code blocks and helper functions from [UNeXt-pytorch](https://github.com/jeya-maria-jose/UNeXt-pytorch). Credit to Jeya Maria Jose for sharing their code implementation.

## Citation
If you find this work useful, please cite the paper below

```bibtex
@article{KIMBOWA2026,
title = {MonoUNet: A Robust Tiny Neural Network for Automated Knee Cartilage Segmentation on Point-of-care Ultrasound Devices},
journal = {Ultrasound in Medicine & Biology},
year = {2026},
issn = {0301-5629},
doi = {https://doi.org/10.1016/j.ultrasmedbio.2026.04.011},
url = {https://www.sciencedirect.com/science/article/pii/S0301562926001572},
author = {Alvin Kimbowa and Arjun Parmar and Ibrahim Mujtaba and Will Wei and Maziar Badii and Matthew Harkey and David Liu and Ilker Hacihaliloglu},
keywords = {Knee cartilage, Segmentation, Ultrasound, Point-of-care ultrasound, Local phase features, Lightweight architecture, Knee osteoarthritis},
}
```