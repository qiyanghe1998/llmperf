#!/usr/bin/env bash
set -euo pipefail

# Resolve repo root (one directory up from this script)
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)

# Activate venv if present
if [ -f "${ROOT}/venv/bin/activate" ]; then
  source "${ROOT}/venv/bin/activate"
fi

# Ensure datasets exist
python "${ROOT}/download_datasets.py"

MODELS=(
  "llama3.1:8b-instruct-q4_K_M"
  "mistral:7b-instruct-q4_K_M"
  "qwen2.5-coder:7b-instruct-q4_K_M"
)

# Process datasets from smallest to largest
DATASETS_ORDERED=(
  "${ROOT}/datasets/prompts_humaneval.txt"   # 164
  "${ROOT}/datasets/prompts_wikitext.txt"    # ~2891
  "${ROOT}/datasets/prompts_mmlu.txt"        # ~14042
)

# Per-dataset settings
max_tokens_for() {
  case "$1" in
    *prompts_wikitext.txt) echo 64 ;;
    *prompts_mmlu.txt) echo 16 ;;
    *prompts_humaneval.txt) echo 320 ;;
    *) echo 128 ;;
  esac
}

temperature_for() {
  case "$1" in
    *prompts_wikitext.txt) echo 0.2 ;;
    *prompts_mmlu.txt) echo 0.1 ;;
    *prompts_humaneval.txt) echo 0.2 ;;
    *) echo 0.2 ;;
  esac
}

timestamp() { date +"%Y%m%d_%H%M%S"; }

# Create results directory and set output directory
mkdir -p "${ROOT}/results"
OUTDIR="${ROOT}/results"

# Helper to run a dataset in batches by writing slice files on the fly
run_dataset_in_batches() {
  local model="$1"
  local ds="$2"
  local batch_size="$3"

  local total
  total=$(wc -l < "$ds" | awk '{print $1}')
  # Apply optional dataset-specific LIMIT_* env caps
  case "$ds" in
    *prompts_humaneval.txt)
      if [ -n "${LIMIT_HUMANEVAL:-}" ] && [ "$LIMIT_HUMANEVAL" -gt 0 ] && [ "$LIMIT_HUMANEVAL" -lt "$total" ]; then
        total="$LIMIT_HUMANEVAL"
      fi
      ;;
    *prompts_wikitext.txt)
      if [ -n "${LIMIT_WIKITEXT:-}" ] && [ "$LIMIT_WIKITEXT" -gt 0 ] && [ "$LIMIT_WIKITEXT" -lt "$total" ]; then
        total="$LIMIT_WIKITEXT"
      fi
      ;;
    *prompts_mmlu.txt)
      if [ -n "${LIMIT_MMLU:-}" ] && [ "$LIMIT_MMLU" -gt 0 ] && [ "$LIMIT_MMLU" -lt "$total" ]; then
        total="$LIMIT_MMLU"
      fi
      ;;
  esac
  local base
  base=$(basename "$ds" .txt)
  local idx=1
  local start=1

  while [ "$start" -le "$total" ]; do
    local end=$(( start + batch_size - 1 ))
    if [ "$end" -gt "$total" ]; then end="$total"; fi
    local slice_file="${ROOT}/datasets/.tmp_${base}_${start}_${end}.txt"
    sed -n "${start},${end}p" "$ds" > "$slice_file"

    local t
    t=$(timestamp)
    local out_json="${OUTDIR}/results_${base}_b${idx}_${start}-${end}_${model//[:\/]/_}_${t}.json"
    echo "Running: model=${model} ${base} batch ${idx} (${start}-${end}/${total}) -> ${out_json}"
    python -m src.cli run \
      --backend ollama \
      --model "${model}" \
      --prompt-file "${slice_file}" \
      --output "${out_json}" \
      --max-tokens "$(max_tokens_for "$ds")" \
      --temperature "$(temperature_for "$ds")"

    rm -f "$slice_file"
    idx=$(( idx + 1 ))
    start=$(( end + 1 ))
  done
}

for model in "${MODELS[@]}"; do
  for ds in "${DATASETS_ORDERED[@]}"; do
    case "$ds" in
      *prompts_humaneval.txt)
        # Smallest; run as a single batch (<= 500)
        run_dataset_in_batches "$model" "$ds" 500
        ;;
      *prompts_wikitext.txt)
        run_dataset_in_batches "$model" "$ds" 500
        ;;
      *prompts_mmlu.txt)
        run_dataset_in_batches "$model" "$ds" 1000
        ;;
    esac
  done
done

echo "All benchmarks completed. Results saved under ${OUTDIR}/results_*.json"


