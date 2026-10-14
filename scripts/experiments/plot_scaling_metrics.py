#!/usr/bin/env python3
"""Plot baseline scaling curves from scaling_metrics.jsonl + optional GPU CSV.

Does not change training. Example:

  python scripts/experiments/plot_scaling_metrics.py \\
      --metrics outputs/experiments/libero_alpha_flow_baseline_<ts>/<run>/scaling_metrics.jsonl \\
      --gpu outputs/logs/gpu_util_<ts>.csv \\
      --out outputs/experiments/libero_alpha_flow_baseline_<ts>/scaling_curves.png
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def _load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--gpu", type=Path, default=None)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = _load_jsonl(args.metrics)
    if not rows:
        raise SystemExit(f"no rows in {args.metrics}")
    steps = [r["step"] for r in rows]
    samples = [r["samples_seen"] for r in rows]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    ax = axes[0, 0]
    ax.plot(steps, [r["loss"] for r in rows], label="total")
    ax.plot(steps, [r["loss_video"] for r in rows], label="video")
    ax.plot(steps, [r["loss_action"] for r in rows], label="action")
    ax.set_xlabel("step")
    ax.set_ylabel("loss")
    ax.set_title("Loss vs step")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.plot(samples, [r["loss"] for r in rows], label="total")
    ax.plot(samples, [r["loss_video"] for r in rows], label="video")
    ax.plot(samples, [r["loss_action"] for r in rows], label="action")
    ax.set_xlabel("samples seen")
    ax.set_ylabel("loss")
    ax.set_title("Loss vs samples (scaling x-axis)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    ax.plot(steps, [r["steps_per_sec"] for r in rows])
    ax.set_xlabel("step")
    ax.set_ylabel("steps/s")
    ax.set_title("Throughput")
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    if args.gpu and args.gpu.is_file():
        with args.gpu.open() as f:
            g = list(csv.DictReader(f))
        by_gpu: dict[str, list] = {}
        for rec in g:
            by_gpu.setdefault(rec["gpu_index"].strip(), []).append(rec)
        for idx, recs in sorted(by_gpu.items(), key=lambda kv: int(kv[0])):
            ax.plot(
                range(len(recs)),
                [float(r["util_gpu"]) for r in recs],
                label=f"gpu{idx}",
                alpha=0.8,
            )
        ax.set_ylabel("GPU util %")
        ax.set_title("GPU utilization")
        ax.legend(ncol=4, fontsize=8)
    else:
        ax.plot(steps, [r["grad_norm"] for r in rows])
        ax.set_ylabel("grad_norm")
        ax.set_title("Gradient norm")
    ax.set_xlabel("sample index")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150)
    print(f"wrote {args.out} n={len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
