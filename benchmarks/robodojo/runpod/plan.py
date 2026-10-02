#!/usr/bin/env python3
"""Resolve an eval config (+ KEY=VALUE overrides) into the launch plan: queue of (task, seed) cells, eval.env, cost.

  plan.py <config.env> [KEY=VALUE ...] [--write DIR]

Prints the plan. With --write it also puts queue.txt, eval.env and plan.json into DIR for launch.sh to upload.
Task names, dimensions, the paired _random siblings and the runtime figures come from ../tasks.json.
"""

import json
import shlex
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
POD_KEYS = ["NAME", "CKPT_DIR", "CKPT_LABEL", "POLICY_DIR", "POLICY_ENV", "RUN_PREFIX", "EVAL_NUM", "MAX_CELL_HOURS"]
DEFAULTS = {
    "TASKS": "all",
    "SEEDS": "0 1 2",
    "EVAL_NUM": "native",
    "PAIR_RANDOM": "1",
    "PODS": "1",
    "DRIVERS": "580.159.04",
    "MAX_CELL_HOURS": "6",
    "CLAIM_LIMIT": "all",
    "RUN_PREFIX": "fw",
    "POLICY_DIR": "XPolicyLab/policy/OpenWAM",
    "POLICY_ENV": "/workspace/oeh-campaign/envs/openwam",
}


def read_config(path, overrides):
    cfg = dict(DEFAULTS)
    for line in Path(path).read_text().splitlines():
        parts = shlex.split(line, comments=True)
        if not parts:
            continue
        if len(parts) != 1 or "=" not in parts[0]:
            sys.exit(f"{path}: not KEY=VALUE: {line}")
        key, value = parts[0].split("=", 1)
        cfg[key] = value
    for item in overrides:
        key, value = item.split("=", 1)
        cfg[key] = value
    for key in ("NAME", "OUT", "CKPT_DIR", "CKPT_LABEL"):
        if not cfg.get(key):
            sys.exit(f"{path}: {key} is required")
    if "/" in cfg["CKPT_LABEL"] or " " in cfg["CKPT_LABEL"]:
        sys.exit("CKPT_LABEL goes into result paths: no slashes or spaces")
    return cfg


def resolve_tasks(spec, pair_random, inv):
    weights = inv["runtime_weight_seconds"]
    paired = set(inv["paired_random"])
    canonical = [t for t in weights if not t.endswith("_random")]
    names = []
    for item in spec.replace(",", " ").split():
        if item == "all":
            names += canonical
        elif item in inv["dimensions"]:
            names += inv["dimensions"][item]
        elif item in weights:
            names.append(item)
        else:
            sys.exit(f"unknown task or dimension: {item} (dimensions: all {' '.join(inv['dimensions'])})")
    dirs = []
    for name in names:
        for d in [name] + ([name + "_random"] if pair_random and name in paired else []):
            if d not in dirs:
                dirs.append(d)
    return dirs


def main():
    args = sys.argv[1:]
    write = None
    if "--write" in args:
        i = args.index("--write")
        write = Path(args[i + 1])
        del args[i : i + 2]
    if not args:
        sys.exit(__doc__)
    cfg = read_config(args[0], args[1:])
    inv = json.loads((HERE.parent / "tasks.json").read_text())
    weights = inv["runtime_weight_seconds"]
    dirs = resolve_tasks(cfg["TASKS"], cfg["PAIR_RANDOM"] == "1", inv)
    seeds = [int(s) for s in cfg["SEEDS"].replace(",", " ").split()]
    cells = sorted(((weights[d], d, s) for d in dirs for s in seeds), key=lambda c: (-c[0], c[1], c[2]))
    factor, rate = inv["pilot_runtime_factor"], inv["gpu_usd_per_hour"]
    native = cfg["EVAL_NUM"] == "native"
    hours = sum(c[0] for c in cells) * factor / 3600
    pods = min(int(cfg["PODS"]), len(cells))
    longest = cells[0][0] * factor / 3600
    wall = max(hours / pods, longest)
    limit = len(cells) if cfg["CLAIM_LIMIT"] == "all" else int(cfg["CLAIM_LIMIT"])
    plan = {
        "config": cfg,
        "task_dirs": len(dirs),
        "seeds": seeds,
        "cells": len(cells),
        "pods": pods,
        "gpu_hours": round(hours, 1),
        "usd": round(hours * rate + pods * 0.05 * rate, 0),
        "wall_hours": round(wall, 1),
        "longest_cell_hours": round(longest, 1),
        "claim_limit_cells": limit,
    }
    print(
        f"{cfg['NAME']}: {len(dirs)} task dirs x seeds {seeds} = {len(cells)} cells on {pods} pod(s), drivers {cfg['DRIVERS']}"
    )
    print(f"  checkpoint {cfg['CKPT_DIR']}  (label {cfg['CKPT_LABEL']})")
    print(f"  results    {cfg['OUT']}  run id {cfg['RUN_PREFIX']}_s<seed>  eval_num {cfg['EVAL_NUM']}")
    if native:
        print(
            f"  estimate   {plan['gpu_hours']} L40S-hours, ~${plan['usd']:.0f}, ~{plan['wall_hours']} h wall "
            f"(longest cell ~{plan['longest_cell_hours']} h); cells already finished under OUT are skipped"
        )
    else:
        print("  estimate   not available for a capped EVAL_NUM (about 10 min start-up per cell plus the episodes)")
    print(f"  spend guard: at most {limit} newly claimed cell(s)")
    if write:
        write.mkdir(parents=True, exist_ok=True)
        (write / "queue.txt").write_text("".join(f"{d} {s}\n" for _, d, s in cells))
        (write / "eval.env").write_text("".join(f"{k}={shlex.quote(cfg[k])}\n" for k in POD_KEYS))
        (write / "plan.json").write_text(json.dumps(plan, indent=1))


if __name__ == "__main__":
    main()
