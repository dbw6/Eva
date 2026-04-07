# EVA Algorithm Evaluation (Optional)

This directory contains scripts to reproduce the algorithm-level accuracy results from the EVA paper (Tables III, IV, and X) using pre-trained AQLM-quantized model checkpoints.

> **Note:** EVA's algorithm (AQLM-based quantization) has **not** been specifically optimized. The primary contribution of this work is the hardware accelerator design. This evaluation is provided as an optional artifact for completeness.

The evaluation code is migrated from the [AQLM repository](https://github.com/Vahe1994/AQLM) with modifications to support Mixture-of-Experts models (Mixtral-8x7B, Qwen3-30B-A3B) and `transformers>=5.x` compatibility.

## Prerequisites

```bash
conda activate eva
cd Eva/   # all commands run from the Eva/ root

pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
pip install "transformers>=5.4.0"
pip install "accelerate>=0.29.3"
pip install "sentencepiece>=0.2.0"
pip install "safetensors>=0.4.0"
pip install "aqlm[gpu,cpu]>=1.1.6"
pip install lm-eval
```

**GPU requirement:** NVIDIA GPU with >=24 GB VRAM (A100-80GB recommended). Set `GPU_ID` to a GPU with low VRAM usage.

## Transformers Compatibility

`transformers>=5.4.0` is required for Qwen3MoE model support. The `model_loader.py` module automatically patches two compatibility issues between `transformers>=5.x` and AQLM checkpoints:

1. **Fused MoE experts**: `transformers>=5.x` uses fused 3D tensors for MoE expert weights, but AQLM checkpoints store per-expert quantization codes/codebooks/scales. The loader replaces fused expert modules with unfused `ModuleList`-based experts that match the AQLM checkpoint layout.
2. **Incorrectly quantized `lm_head`**: `transformers>=5.x` may erroneously apply AQLM quantization to the `lm_head` layer. The loader detects this and replaces it with a standard `nn.Linear` initialized from the checkpoint.

These patches are applied transparently by `eval_ppl.py` and `lmeval.py`.

## HuggingFace Checkpoints

All checkpoints are downloaded automatically from HuggingFace on first run.

| Model | Bits | HF Path | Table |
|-------|------|---------|-------|
| Llama-2-7B  | 4 | `dbw6/Llama-2-7b-AQLM-4Bit-4x8-hf` | III, IV |
| Llama-2-7B  | 2 | `ISTA-DASLab/Llama-2-7b-AQLM-2Bit-2x8-hf` | III, IV |
| Llama-2-13B | 4 | `dbw6/Llama-2-13b-AQLM-4Bit-4x8-hf` | III |
| Llama-2-13B | 2 | `ISTA-DASLab/Llama-2-13b-AQLM-2Bit-2x8-hf` | III |
| Mixtral-8x7B | 4 | `dbw6/Mixtral-8x7B-AQLM-4Bit-4x8-hf` | X |
| Mixtral-8x7B | 2 | `dbw6/Mixtral-8x7B-AQLM-2Bit-2x8-hf` | X |
| Qwen3-30B-A3B | 4 | `dbw6/Qwen3-30B-A3B-Instruct-2507-AQLM-4Bit-4x8-hf` | X |
| Qwen3-30B-A3B | 2 | `dbw6/Qwen3-30B-A3B-Instruct-2507-AQLM-2Bit-2x8-hf` | X |

## Step 10: TABLE III -- WikiText-2 Perplexity

Evaluate perplexity for Llama-2 7B/13B at 4-bit and 2-bit quantization:

```bash
# Llama-2-7B 4-bit
CUDA_VISIBLE_DEVICES=<GPU_ID> python algorithm/eval_ppl.py \
  dbw6/Llama-2-7b-AQLM-4Bit-4x8-hf \
  --output algorithm/output/ppl_llama2_7b_4bit.json

# Llama-2-7B 2-bit
CUDA_VISIBLE_DEVICES=<GPU_ID> python algorithm/eval_ppl.py \
  ISTA-DASLab/Llama-2-7b-AQLM-2Bit-2x8-hf \
  --output algorithm/output/ppl_llama2_7b_2bit.json

# Llama-2-13B 4-bit
CUDA_VISIBLE_DEVICES=<GPU_ID> python algorithm/eval_ppl.py \
  dbw6/Llama-2-13b-AQLM-4Bit-4x8-hf \
  --output algorithm/output/ppl_llama2_13b_4bit.json

# Llama-2-13B 2-bit
CUDA_VISIBLE_DEVICES=<GPU_ID> python algorithm/eval_ppl.py \
  ISTA-DASLab/Llama-2-13b-AQLM-2Bit-2x8-hf \
  --output algorithm/output/ppl_llama2_13b_2bit.json
```

- **Expected runtime**: ~ 5--10 minutes per model (~30 minutes total)
- **Expected outputs**: `algorithm/output/ppl_llama2_{7b,13b}_{2,4}bit.json`

## Step 11: TABLE IV -- Llama-2-7B Downstream Accuracy

```bash
# Llama-2-7B 4-bit
CUDA_VISIBLE_DEVICES=<GPU_ID> PYTHONPATH=algorithm python algorithm/lmeval.py \
  --model hf \
  --model_args pretrained=dbw6/Llama-2-7b-AQLM-4Bit-4x8-hf,dtype=auto,trust_remote_code=True \
  --tasks piqa,copa,arc_easy,arc_challenge,winogrande \
  --batch_size 4 --num_fewshot 0 \
  --output_path algorithm/output/llama2_7b_4bit

# Llama-2-7B 2-bit
CUDA_VISIBLE_DEVICES=<GPU_ID> PYTHONPATH=algorithm python algorithm/lmeval.py \
  --model hf \
  --model_args pretrained=ISTA-DASLab/Llama-2-7b-AQLM-2Bit-2x8-hf,dtype=auto,trust_remote_code=True \
  --tasks piqa,copa,arc_easy,arc_challenge,winogrande \
  --batch_size 4 --num_fewshot 0 \
  --output_path algorithm/output/llama2_7b_2bit
```

- **Expected runtime**: ~ 30--45 minutes per model (~1.5 hours total)
- **Expected outputs**: `algorithm/output/llama2_7b_{2,4}bit/results.json`

## Step 12: TABLE X -- MoE Downstream Accuracy

```bash
# Mixtral-8x7B 4-bit
CUDA_VISIBLE_DEVICES=<GPU_ID> PYTHONPATH=algorithm python algorithm/lmeval.py \
  --model hf \
  --model_args pretrained=dbw6/Mixtral-8x7B-AQLM-4Bit-4x8-hf,dtype=auto,trust_remote_code=True \
  --tasks arc_challenge,arc_easy,piqa,boolq,winogrande \
  --batch_size 4 --num_fewshot 0 \
  --output_path algorithm/output/mixtral_4bit

# Mixtral-8x7B 2-bit
CUDA_VISIBLE_DEVICES=<GPU_ID> PYTHONPATH=algorithm python algorithm/lmeval.py \
  --model hf \
  --model_args pretrained=dbw6/Mixtral-8x7B-AQLM-2Bit-2x8-hf,dtype=auto,trust_remote_code=True \
  --tasks arc_challenge,arc_easy,piqa,boolq,winogrande \
  --batch_size 4 --num_fewshot 0 \
  --output_path algorithm/output/mixtral_2bit

# Qwen3-30B-A3B 4-bit
CUDA_VISIBLE_DEVICES=<GPU_ID> PYTHONPATH=algorithm python algorithm/lmeval.py \
  --model hf \
  --model_args pretrained=dbw6/Qwen3-30B-A3B-Instruct-2507-AQLM-4Bit-4x8-hf,dtype=auto,trust_remote_code=True \
  --tasks arc_challenge,arc_easy,piqa,boolq,winogrande \
  --batch_size 4 --num_fewshot 0 \
  --output_path algorithm/output/qwen3_4bit

# Qwen3-30B-A3B 2-bit
CUDA_VISIBLE_DEVICES=<GPU_ID> PYTHONPATH=algorithm python algorithm/lmeval.py \
  --model hf \
  --model_args pretrained=dbw6/Qwen3-30B-A3B-Instruct-2507-AQLM-2Bit-2x8-hf,dtype=auto,trust_remote_code=True \
  --tasks arc_challenge,arc_easy,piqa,boolq,winogrande \
  --batch_size 4 --num_fewshot 0 \
  --output_path algorithm/output/qwen3_2bit
```

- **Expected runtime**: ~ 45--60 minutes per model (~3--4 hours total)
- **Expected outputs**: `algorithm/output/{mixtral,qwen3}_{2,4}bit/results.json`

## Notebook Visualization

After evaluations complete, launch the algorithm results notebook:

```bash
jupyter notebook notebooks/algorithm_results.ipynb
```

The notebook loads evaluation results from `algorithm/output/`, combines them with baseline data from published papers, and renders the three paper tables. Results that have not yet been computed will appear as blank entries.

## Runtime Summary

All runtimes measured on a single A100-80GB GPU.

| Step | Evaluation | Measured Runtime |
|------|------------|-----------------|
| 10   | TABLE III (4 PPL evals)  | ~30 minutes |
| 11   | TABLE IV (2 downstream)  | ~1.5 hours  |
| 12   | TABLE X (6 downstream)   | ~3--4 hours |
|      | **Total**                | **~5--6 hours** |

## Expected Results

| Paper Artifact | Output File(s) |
|----------------|----------------|
| TABLE III      | `algorithm/output/ppl_llama2_{7b,13b}_{2,4}bit.json` |
| TABLE IV       | `algorithm/output/llama2_7b_{2,4}bit/results.json` |
| TABLE X        | `algorithm/output/{mixtral,qwen3}_{2,4}bit/results.json` |

## Baseline Method References

The following baseline methods appear in the algorithm evaluation tables. Results for these methods are taken from their respective publications:

- **AWQ**: Lin, Ji, et al. "Awq: Activation-aware weight quantization for on-device llm compression and acceleration." Proceedings of machine learning and systems 6 (2024): 87-100.
- **GPTQ**: Frantar, Elias, et al. "Gptq: Accurate post-training quantization for generative pre-trained transformers." arXiv preprint arXiv:2210.17323 (2022).
- **GPTVQ**: Van Baalen, Mart, et al. "Gptvq: The blessing of dimensionality for llm quantization." arXiv preprint arXiv:2402.15319 (2024).
- **BCQ (ShiftAddLLM)**: You, Haoran, et al. "Shiftaddllm: Accelerating pretrained llms via post-training multiplication-less reparameterization." Advances in Neural Information Processing Systems 37 (2024): 24822-24848.
- **AQLM**: Egiazarian, Vage, et al. "Extreme compression of large language models via additive quantization." arXiv preprint arXiv:2401.06118 (2024). Source code: [https://github.com/Vahe1994/AQLM](https://github.com/Vahe1994/AQLM).
- **LLM.265**: Xu, Ceyu, et al. "LLM. 265: Video Codecs are Secretly Tensor Codecs." Proceedings of the 58th IEEE/ACM International Symposium on Microarchitecture. 2025.
- **QServe**: Lin, Yujun, et al. "Qserve: W4a8kv4 quantization and system co-design for efficient llm serving." Proceedings of Machine Learning and Systems 7 (2025).