# EVA: Recasting LLM Decoding into GEMM via an Efficient Vector Quantization Architecture

This repository provides the official implementation and artifacts for the ISCA 2026 paper "EVA: Recasting LLM Decoding into GEMM via an Efficient Vector Quantization Architecture."

This release corresponds to the artifact-evaluated version of the codebase. It includes all scripts, configuration files, and Jupyter notebooks required to reproduce the hardware performance and algorithm accuracy results reported in the paper.

The repository is organized into two main components:

1. **Hardware Simulator** — A config-driven simulator that reproduces all hardware evaluation results (Figures 8--14 and Tables VI--IX). See [`simulator/README.md`](simulator/README.md).
2. **Algorithm Evaluation** — Scripts to reproduce the algorithm-level accuracy tables (Tables III, IV, and X) using pre-trained AQLM-quantized model checkpoints. See [`algorithm/README.md`](algorithm/README.md).

## Artifact Structure

```
Eva/
├── simulator/                   # Hardware simulator (see simulator/README.md)
│   ├── main.py                  # Unified CLI entrypoint
│   ├── configs/                 # YAML study configurations
│   ├── pipelines/               # Per-study simulation pipelines
│   ├── data/                    # Static reference data
│   ├── traces/                  # Pre-processed trace files
│   └── output/                  # Generated results (CSVs, JSONs)
├── algorithm/                   # Algorithm evaluation (see algorithm/README.md)
│   ├── eval_ppl.py              # WikiText-2 perplexity evaluation
│   ├── lmeval.py                # Downstream accuracy evaluation (lm-eval wrapper)
│   ├── model_loader.py          # AQLM model loader with transformers >=5.x compat fixes
│   ├── output/                  # Algorithm evaluation outputs (JSON, git-ignored)
│   └── src/                     # AQLM source (modelutils, datautils, etc.)
├── notebooks/
│   ├── hardware_results.ipynb   # Hardware figure/table reproduction
│   └── algorithm_results.ipynb  # Algorithm table reproduction (optional)
└── pyproject.toml               # Python package definition
```

<!-- ## Study-to-Paper Mapping

| CLI `--study`     | Paper Artifact     | Description                                  |
|-------------------|--------------------|----------------------------------------------|
| `fig9_fc`         | Fig. 9             | FC decode latency and energy                 |
| `fig10_hw`        | Fig. 10 + TABLE VI | Area, power, throughput, efficiency breakdown |
| `fig8_dse`        | Fig. 8             | DSE: EU count and bandwidth sweeps           |
| `table_vii_abl`   | TABLE VII          | Ablation: conflict mitigation, EU scaling    |
| `fig14_index`     | Fig. 14            | Codebook index analysis                      |
| `table_viii_data`  | TABLE VIII         | Dataset sequence length statistics            |
| `table_ix_vq`     | TABLE IX           | VQ configuration normalized latency          |
| `fig11_batch`     | Fig. 11            | Batch scaling on LLaMA-2-7B                  |
| `e2e`             | Fig. 12 + Fig. 13  | End-to-end dense and MoE experiments         | -->

## Hardware and Software Requirements

### Hardware Simulator (Steps 1--9)

- **CPU**: Any x86-64 machine (no GPU required)
- **RAM**: 16 GB or more
- **Disk**: 10+ GB free space

### Algorithm Evaluation (Optional, Steps 10--12)

- **GPU**: NVIDIA GPU with >=24 GB VRAM (A100-80GB recommended)
- **Disk**: ~100 GB additional for model checkpoints
- **CUDA**: 12.x

### Common

- **OS**: Linux (tested on Ubuntu 20.04+)
- **Python**: 3.11
- **Network**: Internet access for HuggingFace model/dataset downloads on first run

## Environment Setup

```bash
conda create -n eva python=3.11 -y
conda activate eva
pip install -e .
pip install "aqlm[gpu,cpu]>=1.1.6"
pip install jupyter nbclient
```

For the optional algorithm evaluation, also install:

```bash
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
pip install "transformers>=5.4.0"
pip install "accelerate>=0.29.3"
pip install "sentencepiece>=0.2.0"
pip install "safetensors>=0.4.0"
pip install lm-eval
```

Notes:
- `aqlm` is required for Table VII, Fig. 14 (hardware), and all algorithm evaluations.
- `datasets` and `transformers` are installed through `pip install -e .`.
- Some studies download gated HuggingFace models on first run. Ensure the environment has access.
- `transformers>=5.4.0` is required for Qwen3MoE model support. The `algorithm/model_loader.py` module automatically patches compatibility issues between `transformers>=5.x` and AQLM checkpoints (see [`algorithm/README.md`](algorithm/README.md) for details).

## Quick Start

```bash
conda activate eva
cd Eva/

# Run all hardware simulation studies (Steps 1--9)
# See simulator/README.md for per-step details
for study in fig9_fc fig10_hw fig8_dse table_vii_abl fig14_index table_viii_data table_ix_vq fig11_batch; do
    python -m simulator.main --study $study --output-dir simulator/output
done
python -m simulator.main --study e2e --output-dir simulator/output \
  --execution-mode full --scenarios fig12_llama2,fig13_moe

# Visualize hardware results
jupyter notebook notebooks/hardware_results.ipynb
```

For the optional algorithm evaluation, see [`algorithm/README.md`](algorithm/README.md).

## Reference

The code for the algorithm section is based on the following repository:

- **AQLM**: Egiazarian, Vage, et al. "Extreme compression of large language models via additive quantization." arXiv preprint arXiv:2401.06118 (2024). Source code: [https://github.com/Vahe1994/AQLM](https://github.com/Vahe1994/AQLM).