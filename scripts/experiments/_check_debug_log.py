#!/usr/bin/env python3
"""Parse an 8-GPU OpenWAM debug log and exit 0 only if the smoke run looks healthy."""
from __future__ import annotations

import argparse
import math
import re
import sys
from pathlib import Path

DEBUG_RE = re.compile(
    r"\[debug\]\[step\s+(\d+)\s+opt\s+(\d+)\]\s+"
    r"loss=([^\s]+)\s+video=([^\s]+)\s+action=([^\s]+)"
    r"(?:.*?mip_t0=([^\s]+)\s+mip_t09=([^\s]+))?"
    r".*?grad_norm=([^\s]+)"
)
FAIL_RE = re.compile(
    r"CUDA out of memory|NCCL error|NCCL WARN|SIGABRT|Traceback \(most recent call last\)|"
    r"Killed process|Watchdog caught|Host OOM|Connection reset",
    re.I,
)


def _finite(token: str) -> bool:
    try:
        v = float(token)
    except ValueError:
        return False
    return math.isfinite(v)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("log", type=Path)
    parser.add_argument("--min-steps", type=int, default=20)
    parser.add_argument("--expect-nproc", type=int, default=int(__import__("os").environ.get("NPROC_PER_NODE", "7")))
    parser.add_argument("--ckpt-dir", type=Path, default=None)
    args = parser.parse_args()

    text = args.log.read_text(errors="replace") if args.log.is_file() else ""
    if not text.strip():
        print("FAIL: empty or missing debug log", file=sys.stderr)
        return 2

    if FAIL_RE.search(text):
        hit = FAIL_RE.search(text).group(0)
        print(f"FAIL: log contains fatal pattern: {hit}", file=sys.stderr)
        return 3

    rows = []
    for m in DEBUG_RE.finditer(text):
        groups = m.groups()
        step, opt, loss, video, action, mip_t0, mip_t09, grad = groups
        vals = {
            "step": int(step),
            "opt": int(opt),
            "loss": loss,
            "video": video,
            "action": action,
            "grad_norm": grad,
        }
        finite_tokens = [loss, video, action, grad]
        if mip_t0 is not None:
            vals["mip_t0"] = mip_t0
            vals["mip_t09"] = mip_t09
            finite_tokens.extend([mip_t0, mip_t09])
        if not all(_finite(v) for v in finite_tokens):
            print(f"FAIL: non-finite metric at step {step}: {vals}", file=sys.stderr)
            return 4
        rows.append(vals)

    if len(rows) < args.min_steps:
        print(f"FAIL: only {len(rows)} finite debug steps, need {args.min_steps}", file=sys.stderr)
        return 5

    n = args.expect_nproc
    nproc_ok = bool(
        re.search(rf"GPUs/node:\s*{n}\b", text) or re.search(rf"world_size[=:]?\s*{n}\b", text, re.I)
    )
    rank_hits = len(set(re.findall(r"Rank\s+(\d+)", text)))
    if not nproc_ok and rank_hits < n:
        print(f"FAIL: did not confirm {n} ranks in the debug log", file=sys.stderr)
        return 6

    ckpts = []
    search_root = args.ckpt_dir if args.ckpt_dir else args.log.parent
    if search_root.exists():
        ckpts = list(search_root.rglob("checkpoint_step_*.safetensors"))
    if not ckpts:
        print("FAIL: debug checkpoint file not found", file=sys.stderr)
        return 7

    last = rows[-1]
    extras = ""
    if "mip_t0" in last:
        extras = f" mip_t0={last['mip_t0']} mip_t09={last['mip_t09']}"
    print(
        "PASS debug: "
        f"steps={len(rows)} last_step={last['step']} "
        f"loss={last['loss']} video={last['video']} action={last['action']} "
        f"grad_norm={last['grad_norm']}{extras} ckpts={len(ckpts)}"
    )
    for p in sorted(ckpts)[-5:]:
        print(f"  ckpt {p} bytes={p.stat().st_size}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
