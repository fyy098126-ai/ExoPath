# ExoPath

[English](README.md) | [简体中文](README_zh.md)

PyTorch implementation of **ExoPath**, a time-series forecasting model with
exogenous-conditioned decomposition and asymmetric trend–residual pathways.

## Overview

ExoPath models trend and residual components through separate forecasting
paths. Historical exogenous variation guides endogenous component formation,
while a low-rank context router refines residual representations.

The design allows exogenous information to play different roles in component
formation and forecasting: it conditions trend extraction, while explicit
contextual routing is applied to the residual pathway. Only historical
observations are required; future exogenous values are not needed.

The model consists of three main components:

- **Exogenous-conditioned decomposition:** adapt endogenous EMA smoothing to
  historical exogenous variation. Exogenous residual context is formed using
  a separate EMA with a fixed smoothing coefficient of 0.5.
- **Channel-independent trend extrapolation:** forecast smooth trend components
  using shared temporal linear mappings and average pooling, applied
  independently to each target channel.
- **Residual-specific contextual routing:** jointly gate the endogenous
  residual and a learnable query, then use a context-axis bottleneck to generate
  a routing mask. The refined query is fused with the gated endogenous
  residual for residual forecasting.

Reversible instance normalization is applied before decomposition. The trend
and residual forecasts are added, then inverse instance normalization restores
the prediction to the input scale.

## Forecasting Settings

- **MS:** predict one endogenous target from its history and historical
  exogenous variables. The target is the last input channel.
- **M:** jointly predict all input channels. EMA rates are learned per channel
  without exogenous conditioning in this setting.

Given input `[B, T, C]`, the output is `[B, H, 1]` for `MS` and
`[B, H, C]` for `M`, where `H` denotes the forecast horizon.

## Environment

The dependency file records the locally checked Python 3.11 / PyTorch 2.1.0
CPU environment. Install the dependencies with:

```bash
pip install -r requirements.txt
```

For GPU training, use a CUDA-enabled PyTorch build. The recorded paper-server
environment is Python 3.11.15, PyTorch 2.11.0+cu128, CUDA 12.8, and an NVIDIA
GeForce RTX 4090; it is separate from the local CPU environment.

## Usage

Run the experiment scripts from the repository root.

**Exogenous forecasting (`MS`):**

```bash
bash scripts/exogenous_forecast/ETTh1/ETTh1_MS_96.sh
```

**Multivariate forecasting (`M`):**

```bash
bash scripts/long_term_forecast/ETTh1/ETTh1_M_96.sh
```

The repository provides 29 `MS` and 28 `M` configurations in `scripts/`.
Adjust the GPU settings before execution. The main arguments are
`--seq_len` (lookback length), `--pred_len` (forecast horizon),
`--d_model` (latent dimension), and `--num_groups` (router bottleneck width).

Training uses Adam and DBLoss by default. DBLoss combines losses on the
decomposed residual and trend components; set `--use_dbloss 0` to use MSE.
Early stopping selects the checkpoint by validation MSE, and evaluation
reports MSE and MAE. Run `python run.py --help` for all available arguments.

## Code Structure

- `models/ExoPath.py`: main model.
- `run.py`: experiment entry point.
- `exp/`: training, validation, and evaluation.
- `data_provider/`: data loading and preprocessing.
- `scripts/`: experiment configurations.
- `utils/`: metrics, learning-rate scheduling, and shared utilities.

See [release notes](GITHUB_RELEASE.md) for current output-path and checkpoint
handling details.

