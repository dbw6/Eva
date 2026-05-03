#!/usr/bin/env bash
set -euo pipefail

# Run EVA optional algorithm evaluations in parallel across GPUs.
#
# Usage:
#   GPU_IDS=0 scripts/run_algorithm_parallel.sh
#   GPU_IDS=0,1,2,3 OUTPUT_DIR=algorithm/output scripts/run_algorithm_parallel.sh
#
# Each GPU runs one evaluation at a time. Jobs are assigned round-robin across
# GPU_IDS, so using multiple GPUs reduces wall-clock time without oversubscribing
# a single GPU.

GPU_IDS="${GPU_IDS:-0}"
OUTPUT_DIR="${OUTPUT_DIR:-algorithm/output}"
LOG_DIR="${LOG_DIR:-${OUTPUT_DIR}/logs/algorithm}"

IFS=',' read -r -a GPUS <<<"${GPU_IDS}"
if (( ${#GPUS[@]} == 0 )); then
  echo "GPU_IDS must contain at least one GPU id, e.g. GPU_IDS=0 or GPU_IDS=0,1" >&2
  exit 2
fi

mkdir -p "${LOG_DIR}" "${OUTPUT_DIR}"

run_job() {
  local gpu="$1"
  local job="$2"
  local log_file="${LOG_DIR}/${job}.log"
  echo "[algorithm][gpu ${gpu}] starting ${job}; log=${log_file}"

  case "${job}" in
    ppl_llama2_7b_4bit)
      CUDA_VISIBLE_DEVICES="${gpu}" python algorithm/eval_ppl.py \
        dbw6/Llama-2-7b-AQLM-4Bit-4x8-hf \
        --output "${OUTPUT_DIR}/ppl_llama2_7b_4bit.json" >"${log_file}" 2>&1
      ;;
    ppl_llama2_7b_2bit)
      CUDA_VISIBLE_DEVICES="${gpu}" python algorithm/eval_ppl.py \
        ISTA-DASLab/Llama-2-7b-AQLM-2Bit-2x8-hf \
        --output "${OUTPUT_DIR}/ppl_llama2_7b_2bit.json" >"${log_file}" 2>&1
      ;;
    ppl_llama2_13b_4bit)
      CUDA_VISIBLE_DEVICES="${gpu}" python algorithm/eval_ppl.py \
        dbw6/Llama-2-13b-AQLM-4Bit-4x8-hf \
        --output "${OUTPUT_DIR}/ppl_llama2_13b_4bit.json" >"${log_file}" 2>&1
      ;;
    ppl_llama2_13b_2bit)
      CUDA_VISIBLE_DEVICES="${gpu}" python algorithm/eval_ppl.py \
        ISTA-DASLab/Llama-2-13b-AQLM-2Bit-2x8-hf \
        --output "${OUTPUT_DIR}/ppl_llama2_13b_2bit.json" >"${log_file}" 2>&1
      ;;
    llama2_7b_4bit)
      CUDA_VISIBLE_DEVICES="${gpu}" PYTHONPATH=algorithm python algorithm/lmeval.py \
        --model hf \
        --model_args pretrained=dbw6/Llama-2-7b-AQLM-4Bit-4x8-hf,dtype=float16,trust_remote_code=True \
        --tasks piqa,copa,arc_easy,arc_challenge,winogrande \
        --batch_size 4 --num_fewshot 0 \
        --output_path "${OUTPUT_DIR}/llama2_7b_4bit" >"${log_file}" 2>&1
      ;;
    llama2_7b_2bit)
      CUDA_VISIBLE_DEVICES="${gpu}" PYTHONPATH=algorithm python algorithm/lmeval.py \
        --model hf \
        --model_args pretrained=ISTA-DASLab/Llama-2-7b-AQLM-2Bit-2x8-hf,dtype=float16,trust_remote_code=True \
        --tasks piqa,copa,arc_easy,arc_challenge,winogrande \
        --batch_size 4 --num_fewshot 0 \
        --output_path "${OUTPUT_DIR}/llama2_7b_2bit" >"${log_file}" 2>&1
      ;;
    mixtral_4bit)
      CUDA_VISIBLE_DEVICES="${gpu}" PYTHONPATH=algorithm python algorithm/lmeval.py \
        --model hf \
        --model_args pretrained=dbw6/Mixtral-8x7B-AQLM-4Bit-4x8-hf,dtype=float16,trust_remote_code=True \
        --tasks arc_challenge,arc_easy,piqa,boolq,winogrande \
        --batch_size 4 --num_fewshot 0 \
        --output_path "${OUTPUT_DIR}/mixtral_4bit" >"${log_file}" 2>&1
      ;;
    mixtral_2bit)
      CUDA_VISIBLE_DEVICES="${gpu}" PYTHONPATH=algorithm python algorithm/lmeval.py \
        --model hf \
        --model_args pretrained=dbw6/Mixtral-8x7B-AQLM-2Bit-2x8-hf,dtype=float16,trust_remote_code=True \
        --tasks arc_challenge,arc_easy,piqa,boolq,winogrande \
        --batch_size 4 --num_fewshot 0 \
        --output_path "${OUTPUT_DIR}/mixtral_2bit" >"${log_file}" 2>&1
      ;;
    qwen3_4bit)
      CUDA_VISIBLE_DEVICES="${gpu}" PYTHONPATH=algorithm python algorithm/lmeval.py \
        --model hf \
        --model_args pretrained=dbw6/Qwen3-30B-A3B-Instruct-2507-AQLM-4Bit-4x8-hf,dtype=float16,trust_remote_code=True \
        --tasks arc_challenge,arc_easy,piqa,boolq,winogrande \
        --batch_size 4 --num_fewshot 0 \
        --output_path "${OUTPUT_DIR}/qwen3_4bit" >"${log_file}" 2>&1
      ;;
    qwen3_2bit)
      CUDA_VISIBLE_DEVICES="${gpu}" PYTHONPATH=algorithm python algorithm/lmeval.py \
        --model hf \
        --model_args pretrained=dbw6/Qwen3-30B-A3B-Instruct-2507-AQLM-2Bit-2x8-hf,dtype=float16,trust_remote_code=True \
        --tasks arc_challenge,arc_easy,piqa,boolq,winogrande \
        --batch_size 4 --num_fewshot 0 \
        --output_path "${OUTPUT_DIR}/qwen3_2bit" >"${log_file}" 2>&1
      ;;
    *)
      echo "Unknown job: ${job}" >&2
      exit 2
      ;;
  esac

  echo "[algorithm][gpu ${gpu}] finished ${job}"
}

JOBS=(
  ppl_llama2_7b_4bit
  ppl_llama2_7b_2bit
  ppl_llama2_13b_4bit
  ppl_llama2_13b_2bit
  llama2_7b_4bit
  llama2_7b_2bit
  mixtral_4bit
  mixtral_2bit
  qwen3_4bit
  qwen3_2bit
)

run_lane() {
  local lane="$1"
  local gpu="${GPUS[$lane]}"
  local idx
  for idx in "${!JOBS[@]}"; do
    if (( idx % ${#GPUS[@]} == lane )); then
      run_job "${gpu}" "${JOBS[$idx]}"
    fi
  done
}

for lane in "${!GPUS[@]}"; do
  run_lane "${lane}" &
done
wait

echo "[algorithm] all evaluations completed; outputs=${OUTPUT_DIR}"
