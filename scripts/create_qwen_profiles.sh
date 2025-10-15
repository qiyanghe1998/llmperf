#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)

profiles=(
  "modelfiles/qwen2.5/fast-q4.Modelfile:qwen2.5-fast-q4"
  "modelfiles/qwen2.5/balanced-q5.Modelfile:qwen2.5-balanced-q5"
  "modelfiles/qwen2.5/quality-q8.Modelfile:qwen2.5-quality-q8"
)

for spec in "${profiles[@]}"; do
  file="${spec%%:*}"
  name="${spec##*:}"
  echo "Creating ${name} from ${file}..."
  ollama create "${name}" -f "${ROOT}/${file}"
done

echo "Done creating qwen2.5 profiles."

