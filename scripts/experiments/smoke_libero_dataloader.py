#!/usr/bin/env python3
"""Data-only smoke test for the official LIBERO training reader.

Does not load the 5B model. Instantiates the training dataset via the same
registry path as scripts/train.py and fetches several batches.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))


def _tensor_report(name: str, value) -> dict:
    if value is None:
        return {"name": name, "type": "None"}
    if isinstance(value, str):
        return {"name": name, "type": "str", "value": value[:120], "len": len(value)}
    if isinstance(value, list):
        first = value[0] if value else None
        shape = getattr(first, "size", None)
        mode = getattr(first, "mode", None)
        return {
            "name": name,
            "type": f"list[{type(first).__name__ if first is not None else 'empty'}]",
            "len": len(value),
            "item_size": list(shape) if shape else None,
            "item_mode": mode,
        }
    if torch.is_tensor(value):
        finite = bool(torch.isfinite(value.float()).all().item()) if value.numel() else True
        return {
            "name": name,
            "type": "tensor",
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "finite": finite,
            "min": float(value.min()) if value.numel() else None,
            "max": float(value.max()) if value.numel() else None,
        }
    if isinstance(value, np.ndarray):
        return {
            "name": name,
            "type": "ndarray",
            "shape": list(value.shape),
            "dtype": str(value.dtype),
        }
    return {"name": name, "type": type(value).__name__}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batches", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "outputs/logs/libero_data_smoke.json")
    args = parser.parse_args()

    from hydra import compose, initialize_config_dir
    from openwam.dataloader.registry import build_dataset

    with initialize_config_dir(config_dir=str(REPO_ROOT / "configs"), version_base=None):
        cfg = compose(config_name="train", overrides=["dataloader=libero"])

    dataset_dir = cfg.dataloader.dataset_dir
    if not Path(dataset_dir).exists() or str(dataset_dir).startswith("/path/to/"):
        raise SystemExit(f"LIBERO dataset_dir is not a real download: {dataset_dir}")

    t0 = time.time()
    dataset = build_dataset(cfg.dataloader, split="train")
    sample = dataset[0]
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=list,
        pin_memory=False,
    )

    batch_reports = []
    rss_mb = []
    try:
        import psutil

        proc = psutil.Process(os.getpid())
    except Exception:
        proc = None

    for i, batch in enumerate(loader):
        if proc is not None:
            rss_mb.append(proc.memory_info().rss / 1e6)
        item = batch[0]
        batch_reports.append(
            {
                "batch_index": i,
                "batch_len": len(batch),
                "fields": [_tensor_report(k, item.get(k)) for k in item],
            }
        )
        if i + 1 >= args.batches:
            break

    report = {
        "dataset_dir": str(dataset_dir),
        "len": len(dataset),
        "num_frames": int(cfg.dataloader.num_frames),
        "video_stride": int(cfg.dataloader.video_stride),
        "action_horizon": int(cfg.dataloader.num_frames) - 1,
        "normalize_mode": str(cfg.dataloader.normalize_mode),
        "unify_action": bool(cfg.dataloader.unify_action),
        "sample0": [_tensor_report(k, sample.get(k)) for k in sample],
        "batches": batch_reports,
        "rss_mb": rss_mb,
        "elapsed_sec": round(time.time() - t0, 2),
        "stats_path": str(Path(dataset_dir) / "meta" / "libero_normalization_stats.npy"),
        "stats_exists": (Path(dataset_dir) / "meta" / "libero_normalization_stats.npy").is_file(),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
