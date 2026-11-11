#!/usr/bin/env python3
# Based on openvino.genai samples/python/text_generation/speculative_decoding_lm.py
# Copyright (C) 2024-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
# Baseline vs MTP vs DFlash benchmark for OpenVINO GenAI 2026.4.
#
# Usage (run from the NeurIPS folder):
#   python spec.py base   models/qwen35-4b
#   python spec.py mtp    models/qwen35-4b --k 2
#   python spec.py dflash models/qwen35-4b --draft models/qwen35-4b-dflash --k 7
# Options: --device GPU|CPU   --tokens 256   --runs 3   --prompt "..."
#
# MTP: the draft is the target folder itself (openvino_mtp_model.xml lives there).
# DFlash: the draft is a separate exported DFlash model folder.
# Both need prefix caching off and greedy decoding.
# Results are appended to results/stats.csv, generated text goes to results/*.txt

import argparse
import csv
import os
import time
from datetime import datetime

import openvino_genai

PROMPTS = {
    "code": "Write a Python function that merges two sorted lists into one sorted list. Include a docstring and a short example.",
    "chat": "Explain to a first-year student why the sky is blue and why sunsets are red.",
    "math": "Solve step by step: A train travels 180 km in 2.5 hours, then 120 km in 1.5 hours. What is its average speed for the whole trip?",
}


def streamer(subword):
    print(subword, end="", flush=True)
    return openvino_genai.StreamingStatus.RUNNING


def main():
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["base", "mtp", "dflash"])
    p.add_argument("model_dir")
    p.add_argument("--draft", default=None, help="DFlash draft folder (dflash mode only)")
    p.add_argument("--k", type=int, default=2, help="num_assistant_tokens (draft length)")
    p.add_argument("--device", default="GPU")
    p.add_argument("--tokens", type=int, default=256, help="max_new_tokens")
    p.add_argument("--runs", type=int, default=3, help="timed runs per prompt")
    p.add_argument("--prompt", default=None, help="custom prompt instead of the 3 built-in ones")
    a = p.parse_args()

    if a.mode == "dflash" and not a.draft:
        p.error("dflash mode needs --draft <dflash model folder>")

    os.makedirs("results", exist_ok=True)
    prompts = {"custom": a.prompt} if a.prompt else PROMPTS

    # Same scheduler for all modes so the comparison is fair.
    sc = openvino_genai.SchedulerConfig()
    sc.enable_prefix_caching = False
    # Qwen3.5/3.6 linear-attention state pool. Needs >= 2 + k rows per sequence,
    # default is too small for k >= 6. 32 covers k up to 15 (and all smaller k).
    # DFlash only: in MTP mode the setting is rejected because the MTP draft graph
    # has no linear-attention cache inputs (default pool still allows MTP k <= 5).
    if a.mode == "dflash":
        sc.num_linear_attention_blocks = 32

    props = {}
    if a.mode == "mtp":
        # MTP: draft_model points at the SAME folder as the target (openvino_mtp_model.xml lives there)
        props["draft_model"] = openvino_genai.draft_model(a.model_dir, a.device)
    elif a.mode == "dflash":
        props["draft_model"] = openvino_genai.draft_model(a.draft, a.device)

    # ContinuousBatchingPipeline + draft_model + prefix caching off is the path
    # confirmed working for Qwen3.5 MTP in the earlier blog tests (LNL, CPU and GPU).
    print(f"Loading {a.mode} on {a.device} with ContinuousBatchingPipeline ...", flush=True)
    t0 = time.time()
    pipe = openvino_genai.ContinuousBatchingPipeline(a.model_dir, sc, a.device, props)
    load_s = time.time() - t0
    print(f"Loaded in {load_s:.1f} s", flush=True)

    config = openvino_genai.GenerationConfig()
    config.max_new_tokens = a.tokens
    config.do_sample = False  # greedy, required for MTP/DFlash and makes outputs comparable
    if a.mode != "base":
        config.num_assistant_tokens = a.k

    # Warmup (first GPU run includes kernel compilation)
    warm = openvino_genai.GenerationConfig()
    warm.max_new_tokens = 16
    warm.do_sample = False
    if a.mode != "base":
        warm.num_assistant_tokens = a.k
    def run_gen(prompt, cfg, stream=None):
        # returns a list with one GenerationResult per prompt
        return pipe.generate([prompt], [cfg], stream)[0]

    run_gen("Hello", warm)

    k_label = a.k if a.mode != "base" else 0
    csv_path = os.path.join("results", "stats.csv")
    new_file = not os.path.exists(csv_path)
    rows = []

    for name, prompt in prompts.items():
        for run in range(a.runs):
            print(f"\n=== {a.mode} k={k_label} | {name} | run {run + 1}/{a.runs} ===", flush=True)
            # Stream text on the first run only, so the demo shows it really working
            res = run_gen(prompt, config, streamer if run == 0 else None)
            print()

            ext = res.extended_perf_metrics
            if ext:
                m = ext.main_model_metrics
                accepted = ext.get_num_accepted_tokens()
                drafted = ext.get_num_draft_tokens()
                acc_rate = 100 * ext.get_draft_acceptance_rate()
                # DFlash in 2026.4.1 reports 0 draft tokens; estimate as k per target step
                if drafted == 0:
                    drafted = len(m.raw_metrics.m_durations) * a.k
                    acc_rate = 100 * accepted / drafted
            else:
                m = res.perf_metrics
                accepted = drafted = 0
                acc_rate = 0.0

            gen_tokens = m.get_num_generated_tokens()
            gen_ms = m.get_generate_duration().mean
            ttft = m.get_ttft().mean
            tpot = m.get_tpot().mean
            steps = len(m.raw_metrics.m_durations) or 1
            tok_s = gen_tokens / (gen_ms / 1000) if gen_ms > 0 else 0.0
            # Tokens produced per target-model step (1.0 for plain decoding)
            accept_len = gen_tokens / steps

            print(f"  tokens={gen_tokens}  throughput={tok_s:.1f} tok/s  TTFT={ttft:.1f} ms  "
                  f"TPOT={tpot:.1f} ms  target steps={steps}  tokens/step={accept_len:.2f}")
            if ext:
                print(f"  draft tokens={drafted}  accepted={accepted}  acceptance rate={acc_rate:.1f}%")

            if run == 0:
                with open(os.path.join("results", f"{a.mode}_k{k_label}_{name}.txt"), "w") as f:
                    f.write(str(res.m_generation_ids[0]))

            rows.append({
                "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "mode": a.mode, "k": k_label, "device": a.device,
                "model": os.path.basename(os.path.normpath(a.model_dir)),
                "draft": os.path.basename(os.path.normpath(a.draft)) if a.draft else "",
                "prompt": name, "run": run + 1, "tokens": gen_tokens,
                "gen_ms": round(gen_ms, 1), "tok_s": round(tok_s, 2),
                "ttft_ms": round(ttft, 1), "tpot_ms": round(tpot, 2),
                "steps": steps, "tokens_per_step": round(accept_len, 2),
                "draft_tokens": drafted, "accepted": accepted,
                "acceptance_pct": round(acc_rate, 1), "load_s": round(load_s, 1),
            })

    with open(csv_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        if new_file:
            w.writeheader()
        w.writerows(rows)

    print("\n===== SUMMARY (mean over runs) =====")
    for name in prompts:
        r = [x for x in rows if x["prompt"] == name]
        n = len(r)
        print(f"{a.mode:6s} k={k_label:<2d} {name:6s}  "
              f"{sum(x['tok_s'] for x in r) / n:7.1f} tok/s  "
              f"TTFT {sum(x['ttft_ms'] for x in r) / n:7.1f} ms  "
              f"tokens/step {sum(x['tokens_per_step'] for x in r) / n:4.2f}  "
              f"accept {sum(x['acceptance_pct'] for x in r) / n:5.1f}%")
    print(f"Saved to {csv_path}")


if __name__ == "__main__":
    main()
