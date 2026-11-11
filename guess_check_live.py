#!/usr/bin/env python3
# Live terminal demo for screen recording: normal decoding vs MTP / DFlash
# on OpenVINO GenAI 2026.4.1. Same pipeline and settings as spec.py
# (ContinuousBatchingPipeline + draft_model, prefix caching off, greedy).
#
# Usage (from the NeurIPS folder, venv active, pip install rich):
#   python guess_check_live.py mtp    models/qwen36-27b --k 2 --compare
#   python guess_check_live.py dflash models/qwen36-27b --draft models/qwen36-27b-dflash --k 7 --compare
# --compare runs normal decoding first, then the chosen mode, and prints the speedup.
# Options: --device GPU|CPU  --tokens 256  --prompt "..."
#
# Live panel: the text grows as the big model emits tokens. The tokens produced
# in the latest step are highlighted, so you can see several tokens land per step
# with MTP / DFlash and one at a time with normal decoding.
# Summary panel: numbers from extended_perf_metrics, computed exactly as in spec.py.

import argparse
import gc
import time

import openvino_genai as ov_genai
from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

PROMPT = ("Write a Python function that merges two sorted lists into one sorted list. "
          "Include a docstring and a short example.")
NAME = {"base": "Normal decoding", "mtp": "MTP", "dflash": "DFlash"}
COLOR = {"base": "bright_blue", "mtp": "yellow", "dflash": "green"}
HIGHLIGHT = {"base": "bold bright_blue", "mtp": "bold black on yellow", "dflash": "bold black on green"}


class LiveStream(ov_genai.StreamerBase):
    """Token-level streamer: write() gets one token, or the list of tokens from one step."""

    def __init__(self, tokenizer, redraw):
        super().__init__()
        self.tok = tokenizer
        self.redraw = redraw
        self.ids = []
        self.steps = 0
        self.last = 0
        self.t_first = None

    def write(self, token):
        new = [int(t) for t in token] if isinstance(token, (list, tuple)) else [int(token)]
        if self.t_first is None:
            self.t_first = time.perf_counter()
        self.ids += new
        self.steps += 1
        self.last = len(new)
        self.redraw()
        return ov_genai.StreamingStatus.RUNNING

    def end(self):
        self.redraw()


def build_pipe(mode, a):
    sc = ov_genai.SchedulerConfig()
    sc.enable_prefix_caching = False
    props = {}
    if mode == "mtp":
        props["draft_model"] = ov_genai.draft_model(a.model_dir, a.device)   # MTP head lives in the target folder
    elif mode == "dflash":
        sc.num_linear_attention_blocks = 32                                   # needed for k >= 6 on Qwen3.5/3.6
        props["draft_model"] = ov_genai.draft_model(a.draft, a.device)       # separate DFlash drafter
    return ov_genai.ContinuousBatchingPipeline(a.model_dir, sc, a.device, props)


def gen_config(mode, a, n):
    c = ov_genai.GenerationConfig()
    c.max_new_tokens = n
    c.do_sample = False
    if mode != "base":
        c.num_assistant_tokens = a.k
    return c


def title(mode, a):
    return f"{NAME[mode]} ({a.device})" + (f" · k = {a.k}" if mode != "base" else "")


def live_panel(mode, a, s, lines):
    if not s.ids:
        body = Text("waiting for the first token ...", style="dim")
    else:
        full = s.tok.decode(s.ids)
        head = s.tok.decode(s.ids[:-s.last]) if s.last < len(s.ids) else ""
        if not full.startswith(head):
            head = full
        body = Text(head) + Text(full[len(head):], style=HIGHLIGHT[mode])
        body = Text("\n").join(body.split("\n")[-lines:])
    n = len(s.ids)
    el = time.perf_counter() - s.t_first if s.t_first else 0.0
    rate = (n - 1) / el if el > 0 and n > 1 else 0.0
    sub = f"{n}/{a.tokens} tokens · {rate:.1f} tok/s · {s.steps} steps"
    if mode != "base" and s.steps:
        sub += f" · {n / s.steps:.2f} tokens/step"
    return Panel(body, title=title(mode, a), subtitle=sub, border_style=COLOR[mode], height=lines + 2)


def note_panel(mode, a, msg, lines):
    return Panel(Text(msg, style="dim"), title=title(mode, a), border_style=COLOR[mode], height=lines + 2)


def stats(res, a):
    ext = res.extended_perf_metrics
    if ext:
        m = ext.main_model_metrics
        accepted = ext.get_num_accepted_tokens()
        drafted = ext.get_num_draft_tokens()
        acc = 100 * ext.get_draft_acceptance_rate()
        if drafted == 0:  # DFlash in 2026.4.1 reports 0 draft tokens: estimate k per target step
            drafted = len(m.raw_metrics.m_durations) * a.k
            acc = 100 * accepted / drafted if drafted else 0.0
    else:
        m = res.perf_metrics
        accepted = drafted = None
        acc = None
    tokens = m.get_num_generated_tokens()
    gen_ms = m.get_generate_duration().mean
    steps = len(m.raw_metrics.m_durations) or 1
    return {
        "tokens": tokens, "steps": steps,
        "tok_s": tokens / (gen_ms / 1000) if gen_ms > 0 else 0.0,
        "ttft": m.get_ttft().mean, "tpot": m.get_tpot().mean,
        "per_step": tokens / steps, "drafted": drafted, "accepted": accepted, "acc": acc,
        "ids": list(res.m_generation_ids[0]),
    }


def summary(modes, R, a):
    t = Table(show_edge=False, header_style="bold")
    t.add_column("Metric")
    for md in modes:
        t.add_column(NAME[md], justify="right", style=COLOR[md])
    fmt = lambda v, f: "-" if v is None else f.format(v)
    rows = [
        ("Throughput (tok/s)", "tok_s", "{:.1f}"), ("TTFT (ms)", "ttft", "{:.0f}"),
        ("TPOT (ms)", "tpot", "{:.1f}"), ("Tokens generated", "tokens", "{}"),
        ("Big-model steps", "steps", "{}"), ("Tokens per step", "per_step", "{:.2f}"),
        ("Guesses drafted", "drafted", "{}"), ("Guesses accepted", "accepted", "{}"),
        ("Acceptance rate (%)", "acc", "{:.1f}"),
    ]
    for label, key, f in rows:
        t.add_row(label, *[fmt(R[md][key], f) for md in modes])
    parts = [t]
    spec = modes[-1]
    if spec != "base":
        parts.append(Text(f"\nnum_assistant_tokens (k) = {a.k}", style="dim"))
        if R[spec]["drafted"] and spec == "dflash":
            parts.append(Text("DFlash drafted count is estimated as k x steps (2026.4.1 reports 0).", style="dim"))
    if "base" in R and spec != "base":
        b, s = R["base"], R[spec]
        parts.append(Text(
            f"\n{NAME[spec]} is {s['tok_s'] / b['tok_s']:.2f}x faster: "
            f"{b['steps']} -> {s['steps']} big-model steps for the same {s['tokens']} tokens",
            style="bold"))
        if a.check_output:
            n = next((i for i, (x, y) in enumerate(zip(b["ids"], s["ids"])) if x != y), min(len(b["ids"]), len(s["ids"])))
            parts.append(Text(f"Token ids match normal decoding for the first {n} of {len(s['ids'])} tokens", style="dim"))
    return Panel(Group(*parts), title=f"Run summary · {a.model_dir}", border_style="white")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["base", "mtp", "dflash"])
    p.add_argument("model_dir")
    p.add_argument("--draft", default=None)
    p.add_argument("--k", type=int, default=2)
    p.add_argument("--device", default="GPU")
    p.add_argument("--tokens", type=int, default=256)
    p.add_argument("--prompt", default=PROMPT)
    p.add_argument("--compare", action="store_true", help="run normal decoding first")
    p.add_argument("--check-output", action="store_true", help="also report where the text first differs")
    a = p.parse_args()
    if a.mode == "dflash" and not a.draft:
        p.error("dflash mode needs --draft <dflash model folder>")
    if a.compare and a.mode == "base":
        p.error("--compare needs mtp or dflash")

    modes = (["base"] if a.compare else []) + [a.mode]
    console = Console()
    lines = max(6, (console.height - 2) // len(modes) - 2)
    panels = {md: note_panel(md, a, "queued", lines) for md in modes}
    R = {}

    with Live(Group(*panels.values()), console=console, refresh_per_second=12) as live:
        show = lambda: live.update(Group(*panels.values()))
        for md in modes:
            panels[md] = note_panel(md, a, "loading model and warming up ...", lines)
            show()
            pipe = build_pipe(md, a)
            pipe.generate([a.prompt], [gen_config(md, a, 16)])  # warm-up: GPU kernel compile
            s = LiveStream(pipe.get_tokenizer(), None)

            def redraw(md=md, s=s):
                panels[md] = live_panel(md, a, s, lines)
                show()

            s.redraw = redraw
            redraw()
            res = pipe.generate([a.prompt], [gen_config(md, a, a.tokens)], s)[0]
            R[md] = stats(res, a)
            redraw()
            del pipe
            gc.collect()

    console.print(summary(modes, R, a))


if __name__ == "__main__":
    main()
