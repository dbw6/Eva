# EVA: Hardware Simulator Artifact

This repository contains the config-driven hardware simulator that reproduces the hardware evaluation results (Figures 8--14 and Tables VI--IX) from the EVA paper.

## Artifact Structure

```
Eva/
├── simulator/
│   ├── main.py                  # Unified CLI entrypoint
│   ├── configs/
│   │   ├── models/              # Dense and MoE model definitions
│   │   ├── methods/             # SA, ANT, FIGNA, FIGLUT, EVA method presets
│   │   ├── studies/             # Named study YAML configs (one per study)
│   │   └── hardware.yaml        # Hardware parameters
│   ├── pipelines/               # Per-study simulation pipelines
│   ├── data/                    # Static reference data (area CSV)
│   ├── traces/                  # Pre-processed trace files
│   └── output/                  # Generated results (CSVs, JSONs)
├── notebooks/
│   └── hardware_results.ipynb   # Visualization and figure reproduction
└── pyproject.toml               # Python package definition
```

## Study-to-Paper Mapping

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
| `e2e`             | Fig. 12 + Fig. 13  | End-to-end dense and MoE experiments         |

## Hardware and Software Requirements

- **CPU**: Any x86-64 machine (no GPU required)
- **RAM**: 16 GB or more
- **Disk**: 10+ GB free space
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

Notes:
- `aqlm` is required for Table VII and Fig. 14 studies, which load AQLM-quantized model weights.
- `datasets` and `transformers` are installed through `pip install -e .`.
- Some studies download gated HuggingFace models on first run. Ensure the environment has access.

## Step-by-Step Reproduction

All commands should be run from the `Eva/` directory with the `eva` conda environment activated. Each study writes its outputs under `simulator/output/<study_name>/`.

### Step 1: Figure 9 -- FC Decode Latency and Energy

```bash
python -m simulator.main --study fig9_fc --output-dir simulator/output
```

- **Expected outputs**: `simulator/output/fig9_fc/aggregated_sequence1.csv`
- **Measured runtime**: 18 seconds

### Step 2: Figure 10 + TABLE VI -- Hardware Characterization

```bash
python -m simulator.main --study fig10_hw --output-dir simulator/output
```

- **Expected outputs**: `simulator/output/fig10_hw/table_vi.csv`, `simulator/output/fig10_hw/fig10_area_breakdown.csv`, `simulator/output/fig10_hw/fig10_power_breakdown.csv`
- **Measured runtime**: 7 seconds

### Step 3: Figure 8 -- Design Space Exploration

```bash
python -m simulator.main --study fig8_dse --output-dir simulator/output
```

- **Expected outputs**: `simulator/output/fig8_dse/fig8_num_eu.csv`, `simulator/output/fig8_dse/fig8_memory_bandwidth_decode.csv`, `simulator/output/fig8_dse/fig8_memory_bandwidth_prefill.csv`
- **Measured runtime**: 9 seconds

### Step 4: TABLE VII -- Ablation

```bash
python -m simulator.main --study table_vii_abl --output-dir simulator/output
```

- **Expected outputs**: `simulator/output/table_vii_abl/table_vii.csv`
- **Measured runtime**: 4 minutes 24 seconds (downloads AQLM Llama-2-7B model weights on first run, ~2 GB; subsequent runs use the cached model)

### Step 5: Figure 14 -- Codebook Index Analysis

```bash
python -m simulator.main --study fig14_index --output-dir simulator/output
```

- **Expected outputs**: `simulator/output/fig14_index/fig14_index_count_histogram_avg.csv`, `simulator/output/fig14_index/fig14_unique_tiles.csv`
- **Measured runtime**: 14 seconds (reuses AQLM model cached from Step 4)

### Step 6: TABLE VIII -- Dataset Statistics

```bash
python -m simulator.main --study table_viii_data --output-dir simulator/output
```

- **Expected outputs**: `simulator/output/table_viii_data/table_viii.csv`
- **Measured runtime**: 28 seconds (downloads HuggingFace datasets and tokenizers on first run)

### Step 7: TABLE IX -- VQ Configurations

```bash
python -m simulator.main --study table_ix_vq --output-dir simulator/output
```

- **Expected outputs**: `simulator/output/table_ix_vq/table_ix.csv`
- **Measured runtime**: 6 seconds

### Step 8: Figure 11 -- Batch Scaling

```bash
python -m simulator.main --study fig11_batch --output-dir simulator/output
```

- **Expected outputs**: `simulator/output/fig11_batch/fig11_batch_scaling.csv`
- **Measured runtime**: 15 seconds

### Step 9: Figure 12 + Figure 13 -- End-to-End

```bash
python -m simulator.main --study e2e --output-dir simulator/output \
  --execution-mode full --scenarios fig12_llama2,fig13_moe
```

End-to-end scenarios:

| Scenario       | Model(s)              | Dataset(s)             | Ops Mode         | Paper Artifact |
|----------------|-----------------------|------------------------|------------------|----------------|
| `fig12_llama2` | Llama-2-7B            | Dolly Creative Writing | FC+Attention+SFU | Fig. 12        |
| `fig13_moe`    | Mixtral-8x7B, Qwen3-30B-A3B | Arxiv, GSM8K    | FC+Attention+SFU | Fig. 13        |

- **Expected outputs**:
  - `simulator/output/e2e/full/fig12_llama2.csv`
  - `simulator/output/e2e/full/fig13_moe.csv`
  - `simulator/output/e2e/full/fig12_llama2/summary.csv`
  - `simulator/output/e2e/full/fig13_moe/summary.csv`
- **Measured runtime**: 1 hour 54 minutes total
  - `fig12_llama2`: ~6 minutes (trace-based, 8 methods)
  - `fig13_moe`: ~1 hour 48 minutes (2 MoE models x 2 datasets x 100 samples x 9 methods)

### Runtime Summary

All runtimes were measured on a single-core x86-64 CPU with 16 GB RAM (no GPU).

| Step | Study          | Measured Runtime       |
|------|----------------|------------------------|
| 1    | `fig9_fc`      | 18 seconds             |
| 2    | `fig10_hw`     | 7 seconds              |
| 3    | `fig8_dse`     | 9 seconds              |
| 4    | `table_vii_abl`| 4 min 24 sec           |
| 5    | `fig14_index`  | 14 seconds             |
| 6    | `table_viii_data`| 28 seconds           |
| 7    | `table_ix_vq`  | 6 seconds              |
| 8    | `fig11_batch`  | 15 seconds             |
| 9    | `e2e`          | 1 hour 54 min          |
|      | **Total**      | **~2 hours**           |

The total runtime is dominated by Step 9 (`e2e`), specifically the `fig13_moe` scenario. Steps 1--3 and 5--8 complete in under 5 minutes combined. Step 4 (`table_vii_abl`) takes several minutes due to downloading the AQLM model on first run.

## Notebook Visualization

After all CLI studies complete, launch the notebook to visualize results and reproduce paper figures:

```bash
jupyter notebook notebooks/hardware_results.ipynb
```

Run the notebook cells from top to bottom. Each section corresponds to one paper figure or table. The notebook reads CSV outputs from `simulator/output/` and renders the paper-facing tables and plots.

## Expected Results

The following output files correspond to the paper artifacts:

| Paper Artifact | Output File(s)                                                |
|----------------|---------------------------------------------------------------|
| TABLE VI       | `simulator/output/fig10_hw/table_vi.csv`                      |
| TABLE VII      | `simulator/output/table_vii_abl/table_vii.csv`                |
| TABLE VIII     | `simulator/output/table_viii_data/table_viii.csv`              |
| TABLE IX       | `simulator/output/table_ix_vq/table_ix.csv`                   |
| Fig. 8         | `simulator/output/fig8_dse/fig8_num_eu.csv`, `fig8_memory_bandwidth_*.csv` |
| Fig. 9         | `simulator/output/fig9_fc/aggregated_sequence1.csv`           |
| Fig. 10        | `simulator/output/fig10_hw/fig10_area_breakdown.csv`, `fig10_power_breakdown.csv` |
| Fig. 11        | `simulator/output/fig11_batch/fig11_batch_scaling.csv`        |
| Fig. 12        | `simulator/output/e2e/full/fig12_llama2.csv`                  |
| Fig. 13        | `simulator/output/e2e/full/fig13_moe.csv`                     |
| Fig. 14        | `simulator/output/fig14_index/fig14_index_count_histogram_avg.csv` |
