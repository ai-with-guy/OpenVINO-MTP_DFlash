#!/usr/bin/env bash
# Downloads the target models and DFlash drafters into models/
set -e
mkdir -p models
dl() { hf download "$1" --local-dir "models/$2"; }
dl OpenVINO/Qwen3.5-4B-int4-ov                 qwen35-4b
dl circulus/Qwen3.5-4B-DFlash-ov-int4          qwen35-4b-dflash
dl OpenVINO/Qwen3.6-27B-int4-ov                qwen36-27b
dl circulus/Qwen3.6-27B-DFlash-ov-int4         qwen36-27b-dflash
dl OpenVINO/Qwen3.6-35B-A3B-int4-ov            qwen36-35b
dl circulus/Qwen3.6-35B-A3B-DFlash-ov-int4     qwen36-35b-dflash
