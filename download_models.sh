#!/usr/bin/env bash
# Downloads models into models/
#   bash download_models.sh                       all six folders
#   bash download_models.sh qwen35-4b qwen35-4b-dflash   only the folders you name
set -e
declare -A REPO=(
  [qwen35-4b]=OpenVINO/Qwen3.5-4B-int4-ov
  [qwen35-4b-dflash]=circulus/Qwen3.5-4B-DFlash-ov-int4
  [qwen36-27b]=OpenVINO/Qwen3.6-27B-int4-ov
  [qwen36-27b-dflash]=circulus/Qwen3.6-27B-DFlash-ov-int4
  [qwen36-35b]=OpenVINO/Qwen3.6-35B-A3B-int4-ov
  [qwen36-35b-dflash]=circulus/Qwen3.6-35B-A3B-DFlash-ov-int4
)
ORDER=(qwen35-4b qwen35-4b-dflash qwen36-27b qwen36-27b-dflash qwen36-35b qwen36-35b-dflash)
NAMES=("$@"); [ ${#NAMES[@]} -eq 0 ] && NAMES=("${ORDER[@]}")
mkdir -p models
for n in "${NAMES[@]}"; do
  [ -z "${REPO[$n]}" ] && { echo "Unknown folder: $n (choose from: ${ORDER[*]})"; exit 1; }
  echo "Downloading ${REPO[$n]} -> models/$n"
  hf download "${REPO[$n]}" --local-dir "models/$n"
done
