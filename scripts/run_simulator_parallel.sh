#!/usr/bin/env bash
set -euo pipefail

# Run EVA hardware simulator studies with bounded parallelism.
#
# Usage:
#   scripts/run_simulator_parallel.sh
#   MAX_JOBS=6 OUTPUT_DIR=simulator/output scripts/run_simulator_parallel.sh
#
# Independent short studies run in parallel. The AQLM-backed studies run
# afterward in order to avoid duplicate first-run checkpoint downloads. The
# end-to-end study then runs once because it writes aggregate files shared by
# its scenarios.

OUTPUT_DIR="${OUTPUT_DIR:-simulator/output}"
MAX_JOBS="${MAX_JOBS:-4}"
LOG_DIR="${LOG_DIR:-${OUTPUT_DIR}/logs/simulator}"

PARALLEL_STUDIES=(
  fig9_fc
  fig10_hw
  fig8_dse
  table_viii_data
  table_ix_vq
  fig11_batch
)

mkdir -p "${LOG_DIR}"

run_study() {
  local study="$1"
  local log_file="${LOG_DIR}/${study}.log"
  echo "[simulator] starting ${study}; log=${log_file}"
  python -m simulator.main --study "${study}" --output-dir "${OUTPUT_DIR}" >"${log_file}" 2>&1
  echo "[simulator] finished ${study}"
}

active_jobs=0
for study in "${PARALLEL_STUDIES[@]}"; do
  run_study "${study}" &
  active_jobs=$((active_jobs + 1))
  if (( active_jobs >= MAX_JOBS )); then
    wait -n
    active_jobs=$((active_jobs - 1))
  fi
done
wait

run_study table_vii_abl
run_study fig14_index

echo "[simulator] starting e2e; log=${LOG_DIR}/e2e.log"
python -m simulator.main --study e2e --output-dir "${OUTPUT_DIR}" \
  --execution-mode full --scenarios fig12_llama2,fig13_moe >"${LOG_DIR}/e2e.log" 2>&1
echo "[simulator] finished e2e"

echo "[simulator] all studies completed; outputs=${OUTPUT_DIR}"
