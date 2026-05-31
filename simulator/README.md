# EVA Hardware Simulator

This directory contains the config-driven hardware simulator that reproduces all hardware evaluation results from the EVA paper (Figures 8--14 and Tables III, VIII, IX, and X).

## Overview

The simulator models cycle-accurate latency, energy, power, and area for EVA and baseline architectures (SA, ANT, FIGNA, FIGLUT) across dense and Mixture-of-Experts LLMs.

## Prerequisites

```bash
conda activate eva
cd Eva/   # all commands run from the Eva/ root
```

No GPU is required. All simulations run on the CPU.

The artifact was tested with GCC/G++ 11.4.0. The simulator itself does not compile native kernels, but Step 4 and Step 5 use AQLM metadata and may require AQLM to be installed correctly in the same environment.

## Parallel Reproduction Script

To run the complete hardware simulation workflow with bounded parallelism:

```bash
scripts/run_simulator_parallel.sh
```

The script runs independent short studies in parallel, runs the AQLM-backed Step 4 and Step 5 in order to avoid duplicate first-run checkpoint downloads, and then runs Step 9 once because the end-to-end scenarios write shared aggregate files. Optional environment variables:

```bash
MAX_JOBS=6 OUTPUT_DIR=simulator/output scripts/run_simulator_parallel.sh
```

Logs are written to `simulator/output/logs/simulator/` by default. The step-by-step commands below remain useful for debugging or reproducing one figure/table at a time.

## Step-by-Step Reproduction

### Step 1: Figure 10 -- FC Decode Latency and Energy

```bash
python -m simulator.main --study fig10_fc --output-dir simulator/output
```

- **Expected outputs**: `simulator/output/fig10_fc/aggregated_sequence1.csv`
- **Measured runtime**: 18 seconds

### Step 2: Figure 9 + TABLE VIII -- Hardware Characterization

```bash
python -m simulator.main --study fig9_hw --output-dir simulator/output
```

- **Expected outputs**: `simulator/output/fig9_hw/table_viii.csv`, `simulator/output/fig9_hw/fig9_area_breakdown.csv`, `simulator/output/fig9_hw/fig9_power_breakdown.csv`
- **Measured runtime**: 7 seconds

### Step 3: Figure 8 -- Design Space Exploration

```bash
python -m simulator.main --study fig8_dse --output-dir simulator/output
```

- **Expected outputs**: `simulator/output/fig8_dse/fig8_num_eu.csv`, `simulator/output/fig8_dse/fig8_memory_bandwidth_decode.csv`, `simulator/output/fig8_dse/fig8_memory_bandwidth_prefill.csv`
- **Measured runtime**: 9 seconds

### Step 4: TABLE X -- Ablation

```bash
python -m simulator.main --study table_x_abl --output-dir simulator/output
```

- **Expected outputs**: `simulator/output/table_x_abl/table_x.csv`
- **Measured runtime**: 4 minutes 24 seconds (downloads AQLM Llama-2-7B model weights on first run, ~2 GB; subsequent runs use the cached model)

### Step 5: Figure 14 -- Codebook Index Analysis

```bash
python -m simulator.main --study fig14_index --output-dir simulator/output
```

- **Expected outputs**: `simulator/output/fig14_index/fig14_index_count_histogram_avg.csv`, `simulator/output/fig14_index/fig14_unique_tiles.csv`
- **Measured runtime**: 14 seconds (reuses AQLM model cached from Step 4)

### Step 6: TABLE IX -- Dataset Statistics

```bash
python -m simulator.main --study table_ix_data --output-dir simulator/output
```

- **Expected outputs**: `simulator/output/table_ix_data/table_ix.csv`
- **Measured runtime**: 28 seconds (downloads HuggingFace datasets and tokenizers on first run)

### Step 7: TABLE III -- VQ Configurations

```bash
python -m simulator.main --study table_iii_vq --output-dir simulator/output
```

- **Expected outputs**: `simulator/output/table_iii_vq/table_iii.csv`
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

## Runtime Summary

All runtimes were measured on a single-core x86-64 CPU with 16 GB RAM (no GPU).

| Step | Study          | Measured Runtime       |
|------|----------------|------------------------|
| 1    | `fig10_fc`     | 18 seconds             |
| 2    | `fig9_hw`      | 7 seconds              |
| 3    | `fig8_dse`     | 9 seconds              |
| 4    | `table_x_abl`  | 4 min 24 sec           |
| 5    | `fig14_index`  | 14 seconds             |
| 6    | `table_ix_data`| 28 seconds             |
| 7    | `table_iii_vq` | 6 seconds              |
| 8    | `fig11_batch`  | 15 seconds             |
| 9    | `e2e`          | 1 hour 54 min          |
|      | **Total**      | **~2 hours**           |

The total runtime is dominated by Step 9 (`e2e`), specifically the `fig13_moe` scenario. Steps 1--3 and 5--8 complete in under 5 minutes combined.

## Notebook Visualization

After all CLI studies complete, launch the notebook to visualize results and reproduce paper figures:

```bash
jupyter notebook notebooks/hardware_results.ipynb
```

Run the notebook cells from top to bottom. Each section corresponds to one paper figure or table. The notebook reads CSV outputs from `simulator/output/` and renders the paper-facing tables and plots.

## Expected Results

| Paper Artifact | Output File(s)                                                |
|----------------|---------------------------------------------------------------|
| TABLE III      | `simulator/output/table_iii_vq/table_iii.csv`                 |
| TABLE VIII     | `simulator/output/fig9_hw/table_viii.csv`                     |
| TABLE IX       | `simulator/output/table_ix_data/table_ix.csv`                 |
| TABLE X        | `simulator/output/table_x_abl/table_x.csv`                    |
| Fig. 8         | `simulator/output/fig8_dse/fig8_num_eu.csv`, `fig8_memory_bandwidth_*.csv` |
| Fig. 9         | `simulator/output/fig9_hw/fig9_area_breakdown.csv`, `fig9_power_breakdown.csv` |
| Fig. 10        | `simulator/output/fig10_fc/aggregated_sequence1.csv`          |
| Fig. 11        | `simulator/output/fig11_batch/fig11_batch_scaling.csv`        |
| Fig. 12        | `simulator/output/e2e/full/fig12_llama2.csv`                  |
| Fig. 13        | `simulator/output/e2e/full/fig13_moe.csv`                     |
| Fig. 14        | `simulator/output/fig14_index/fig14_index_count_histogram_avg.csv` |

## Experiment Customization

Users can customize experiments by editing the YAML configuration files under `configs/studies/`. Each study YAML specifies models, methods, sequence lengths, batch sizes, and study-specific parameters. The CLI also accepts overrides via command-line arguments (e.g., `--models`, `--methods`, `--scenarios`).
