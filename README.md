# OpenVINO™ MTP and DFlash on AI PC

Run Multi-Token Prediction (MTP) and DFlash speculative decoding with OpenVINO™ GenAI on an AI PC with an Intel® Core™ Ultra processor, and measure how much faster each one generates text than normal decoding.

## What this does

Normal decoding produces one token per pass of the model. Speculative decoding guesses several tokens ahead, and the model checks all the guesses in a single pass, keeping the ones that match. The output follows the same greedy decoding, with fewer passes of the big model.

MTP uses the small MTP module that ships inside the model folder. DFlash uses a separate small drafter. The code is the same for both; only the `draft_model` path changes.

## Tested on

| | |
|---|---|
| Processor | Intel® Core™ Ultra Series 3 processor with integrated Intel® Arc™ GPU |
| Memory | 64 GB |
| Operating system | Ubuntu 24.04 |
| Software | OpenVINO™ GenAI 2026.4.1, Python 3.12 |

## Requirements

- An AI PC with an Intel® Core™ Ultra processor and its integrated Intel® Arc™ GPU
- Ubuntu 24.04 with the Intel GPU driver installed. Follow [Configurations for Intel® Processor Graphics (GPU) with OpenVINO™](https://docs.openvino.ai/2026/get-started/install-openvino/configurations/configurations-intel-gpu.html)
- Python 3.10 to 3.13

## Quick start (Qwen3.5-4B)

```bash
# 1. Get the code
git clone https://github.com/ai-with-guy/OpenVINO-MTP_DFlash.git
cd OpenVINO-MTP_DFlash

# 2. Create a Python environment and install
python3 -m venv ov
source ov/bin/activate
pip install -r requirements.txt

# 3. Check that OpenVINO sees the GPU (expect ['CPU', 'GPU', 'NPU'])
python -c "import openvino as ov; print(ov.Core().available_devices)"

# 4. Download the 4B model and its DFlash drafter
bash download_models.sh qwen35-4b qwen35-4b-dflash
# Different model, for example Qwen3.6-27B:
# bash download_models.sh qwen36-27b qwen36-27b-dflash

# 5. Run normal decoding, then MTP, then DFlash
python spec.py base   models/qwen35-4b
python spec.py mtp    models/qwen35-4b --k 6
python spec.py dflash models/qwen35-4b --draft models/qwen35-4b-dflash --k 5
# Different model, for example Qwen3.6-27B:
# python spec.py base   models/qwen36-27b
# python spec.py mtp    models/qwen36-27b --k 6
# python spec.py dflash models/qwen36-27b --draft models/qwen36-27b-dflash --k 7
```

Each command prints a summary table and appends one row per run to `results/stats.csv`. Compare `tok_s` between `base`, `mtp` and `dflash` to see the speedup.

## Live demo: guess_check_live.py

```bash
python guess_check_live.py mtp    models/qwen35-4b --k 6 --compare
python guess_check_live.py dflash models/qwen35-4b --draft models/qwen35-4b-dflash --k 5 --compare
# Different model, for example Qwen3.6-27B:
# python guess_check_live.py dflash models/qwen36-27b --draft models/qwen36-27b-dflash --k 7 --compare
```

`--compare` runs normal decoding first and then MTP or DFlash in a second panel, and ends with a summary table and the speedup. The highlighted text in each panel is what the latest step produced, so you can see several tokens arrive at once with MTP and DFlash.
