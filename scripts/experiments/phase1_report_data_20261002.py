#!/usr/bin/env python3
"""Print one phase-1 arm's report data as compact JSON (shallow-head plan, 2026-10-02).

Run on h100-box from the training tree; the output is small enough to relay in one message:

    python3 scripts/experiments/phase1_report_data_20261002.py l1 [--ref A=<run dir> ...]

It bundles the arm's key config, its training curve binned into at most --bins points, its
checkpoints, the queue's status lines for it, and the summaries of its LIBERO evals. Each
--ref adds a reference run's binned curve (read-only), e.g. A or C for the LIBERO pair.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import re

import yaml

LOG_DIR = "logs/phase1-20261002"
RUNS = {
    "l1": ("libero_dot_l1_20261002", ["l1_fm10", "l1_fm2"]),
    "l2": ("libero_dot_l2_20261002", ["l2_mip"]),
    "r1": ("robodojo_alpha_r1_20261002", []),
    "r3": ("robodojo_alpha_r3_20261002", []),
    "r4": ("robodojo_alpha_r4_20261002", []),
    "r5": ("robodojo_alpha_r5_20261002", []),
}
CURVE_KEYS = (
    "loss",
    "loss_video",
    "loss_action",
    "loss_mip_t0",
    "loss_mip_t09",
    "grad_norm",
    "lr",
    "xm_alt_fraction",
)
CONFIG_SECTIONS = ("training", "model.architecture", "model.action_backbone", "dataloader", "project")
PATH_HINTS = ("path", "dir", "output", "wandb")


def short(x: float) -> float:
    return float(f"{x:.5g}")


def flatten(node, prefix=""):
    out = {}
    if isinstance(node, dict):
        for key, value in node.items():
            out.update(flatten(value, f"{prefix}{key}."))
    else:
        out[prefix[:-1]] = node
    return out


def key_config(run_dir: str) -> dict:
    with open(os.path.join(run_dir, "config.yaml"), encoding="utf-8") as f:
        flat = flatten(yaml.safe_load(f) or {})
    keep = {}
    for key, value in flat.items():
        if not key.startswith(CONFIG_SECTIONS) or any(h in key.split(".")[-1] for h in PATH_HINTS):
            continue
        if isinstance(value, (list, dict)) and len(json.dumps(value)) > 200:
            continue
        keep[key] = value
    return keep


def binned_curve(run_dir: str, bins: int) -> dict:
    rows = []
    path = os.path.join(run_dir, "scaling_metrics.jsonl")
    if os.path.isfile(path):
        with open(path, encoding="utf-8") as f:
            for line in f:
                try:
                    rows.append(json.loads(line))
                except ValueError:
                    continue
    # A resumed run can repeat steps; keep the last row written for each step.
    by_step = {}
    for row in rows:
        by_step[row["step"]] = row
    rows = [by_step[s] for s in sorted(by_step)]
    if not rows:
        return {"points": [], "rows": 0}
    size = max(1, math.ceil(len(rows) / bins))
    points = []
    for start in range(0, len(rows), size):
        chunk = rows[start : start + size]
        point = {"step": chunk[-1]["step"]}
        for key in CURVE_KEYS:
            vals = [r[key] for r in chunk if isinstance(r.get(key), (int, float)) and math.isfinite(r[key])]
            if vals:
                point[key] = short(sum(vals) / len(vals))
        if len(chunk) > 1 and chunk[-1]["step"] > chunk[0]["step"]:
            point["sec_per_step"] = short((chunk[-1]["ts"] - chunk[0]["ts"]) / (chunk[-1]["step"] - chunk[0]["step"]))
        points.append(point)
    nonfinite = sum(
        1
        for r in rows
        for k in ("loss", "loss_video", "loss_action")
        if isinstance(r.get(k), float) and not math.isfinite(r[k])
    )
    return {
        "rows": len(rows),
        "first_ts": rows[0]["ts"],
        "last_ts": rows[-1]["ts"],
        "last_step": rows[-1]["step"],
        "last_opt_step": rows[-1].get("opt_step"),
        "last_epoch": rows[-1].get("epoch"),
        "global_batch": rows[-1].get("global_batch"),
        "nonfinite_losses": nonfinite,
        "points": points,
    }


def eval_summary(tag: str) -> dict | None:
    path = f"outputs/libero/phase1_20261002/{tag}/summary.json"
    if not os.path.isfile(path):
        return None
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    tasks = [
        {k: r.get(k) for k in ("suite", "task_id", "task_name", "successes", "trials")}
        for r in data.get("task_results", [])
    ]
    return {
        "overall": data.get("overall"),
        "suites": data.get("suite_results"),
        "tasks": tasks,
        "missing": len(data.get("missing_runs", [])),
        "mtime": os.path.getmtime(path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("arm", choices=sorted(RUNS))
    parser.add_argument("--bins", type=int, default=100)
    parser.add_argument("--ref", action="append", default=[], help="NAME=RUN_DIR of a reference run")
    args = parser.parse_args()

    name, eval_tags = RUNS[args.arm]
    dirs = sorted(d for d in glob.glob(f"outputs/openwam_checkpoints/{name}/*/") if os.path.isfile(d + "config.yaml"))
    if not dirs:
        raise SystemExit(f"no run dir for {name}")
    run_dir = dirs[-1].rstrip("/")
    ckpts = []
    for path in glob.glob(run_dir + "/checkpoint_step_*.safetensors"):
        step = int(re.findall(r"\d+", os.path.basename(path))[-1])
        ckpts.append({"step": step, "gb": round(os.path.getsize(path) / 1e9, 1), "mtime": os.path.getmtime(path)})
    status = []
    status_path = os.path.join(LOG_DIR, "status.txt")
    if os.path.isfile(status_path):
        with open(status_path, encoding="utf-8") as f:
            status = [ln.rstrip() for ln in f if name in ln or f"train_{args.arm}" in ln or "START queue" in ln]
    bundle = {
        "arm": args.arm,
        "run_name": name,
        "run_dir": os.path.abspath(run_dir),
        "config": key_config(run_dir),
        "curve": binned_curve(run_dir, args.bins),
        "checkpoints": sorted(ckpts, key=lambda c: c["step"]),
        "status": status[-40:],
        "evals": {tag: eval_summary(tag) for tag in eval_tags},
        "refs": {},
    }
    for ref in args.ref:
        ref_name, _, ref_dir = ref.partition("=")
        bundle["refs"][ref_name] = binned_curve(ref_dir, args.bins)
    print(json.dumps(bundle, separators=(",", ":")))


if __name__ == "__main__":
    main()
