# WaveFront Decoding

We introduce Wavefront Decoding (WFD), a training-free,
lossless self-speculative decoding framework native to looped language models.

**Paper**:
[![arXiv](https://img.shields.io/badge/arXiv-2609.23033-b31b1b.svg)](https://arxiv.org/abs/2609.23033)

## Environment Setup

Make sure you pulled submodules
```bash
git submodule update --init
```

We recommend creating a virtual environment and installing PyTorch first.

```bash
conda create -n [name] python=3.11 -y
conda activate [name]
pip install torch
pip install -r env/requirements.txt

python env/verify_env.py
```

## Spec-Bench Protocol

```bash
CUDA_VISIBLE_DEVICES=[gpu_num] bash experiments/exp2_spec_bench/run.sh
```

Edit `run.sh` to configure options before running.

## MATH-500 and GSM8K

```bash
CUDA_VISIBLE_DEVICES=[gpu_num] bash experiments/exp3_accuracy/run.sh
```

Edit `run.sh` to configure options before running.

## Acceptance-Controlled Evaluation

```bash
CUDA_VISIBLE_DEVICES=[gpu_num] bash experiments/exp4_emu/run.sh
```

Edit `run.sh` to configure options before running.

## CUDA Graph Evaluation

```bash
CUDA_VISIBLE_DEVICES=[gpu_num] bash experiments/exp5_graph/run.sh
```

Edit `run.sh` to configure options before running.
