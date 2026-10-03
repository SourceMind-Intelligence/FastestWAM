#!/usr/bin/env python3
"""Summarize the L2 diagnostics run by scripts/experiments/l2_diag_20261003.sh.

For each run (l2_mip, l2_pass1, l1_fm2) and task it reports:

- successes out of 50, and the median steps of the successful trials;
- motion statistics from the recordings, split by outcome:
  - end-effector path length per 100 steps;
  - the share of steps whose xyz command reverses the previous one;
  - gripper command switches per trial;
  - the share of steps where the end effector barely moves;
- from the traces, how much the second MIP pass changes the first (RMS over the used action dims).

It also draws contact sheets for a few trials that L2 failed and L1 solved: one PNG per trial, one row
per run, eight agent-view frames spread over each episode. Only numpy is needed.

Usage: l2_diag_summary_20261003.py [diag dir]
(default outputs/libero/phase1_20261002/diag_20261003). It prints a short text report and writes
summary.json and sheets/*.png into the diag dir.
"""

from __future__ import annotations

import json
import struct
import sys
import zlib
from pathlib import Path

import numpy as np

RUNS = ("l2_mip", "l2_pass1", "l1_fm2")
SHEET_FRAMES = 8
SHEETS_PER_TASK = 3
REVERSAL_MIN_NORM = 0.05  # xyz commands smaller than this (in LIBERO action units) count as no command
STILL_PER_STEP_M = 3e-4  # end-effector moves under 0.3 mm in a step count as still


def write_png(path: Path, image: np.ndarray) -> None:
    """Write an RGB uint8 image (H, W, 3) as a PNG."""
    image = np.ascontiguousarray(image, dtype=np.uint8)
    height, width, _ = image.shape
    raw = b"".join(b"\x00" + image[row].tobytes() for row in range(height))

    def chunk(kind: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)

    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(
        b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header) + chunk(b"IDAT", zlib.compress(raw, 6)) + chunk(b"IEND", b"")
    )


def motion_stats(actions: np.ndarray, eef_pos: np.ndarray) -> dict:
    """Per-trial motion statistics from the per-step actions (T, 7) and end-effector positions (T + 1, 3)."""
    steps = len(actions)
    stats = {"steps": steps}
    if steps < 2 or len(eef_pos) < 2:
        return stats
    moves = np.linalg.norm(np.diff(eef_pos, axis=0), axis=1)
    stats["path_per_100_steps_m"] = float(np.nansum(moves) / steps * 100)
    stats["still_share"] = float(np.mean(moves < STILL_PER_STEP_M))
    xyz = actions[:, :3]
    norms = np.linalg.norm(xyz, axis=1)
    pairs = (norms[1:] > REVERSAL_MIN_NORM) & (norms[:-1] > REVERSAL_MIN_NORM)
    if pairs.any():
        cosine = np.sum(xyz[1:] * xyz[:-1], axis=1) / np.maximum(norms[1:] * norms[:-1], 1e-9)
        stats["reversal_share"] = float(np.mean(cosine[pairs] < 0))
    grip = np.sign(actions[:, 6])
    grip = grip[grip != 0]
    stats["gripper_switches"] = int(np.count_nonzero(np.diff(grip))) if len(grip) > 1 else 0
    return stats


def mean_of(rows: list[dict], key: str) -> float | None:
    values = [row[key] for row in rows if key in row]
    return float(np.mean(values)) if values else None


def load_results(task_dir: Path) -> dict[int, dict]:
    trials = {}
    for path in sorted(task_dir.glob("videos/**/results.json")):
        for trial in json.loads(path.read_text(encoding="utf-8")).get("trials", []):
            trials[int(trial["trial"])] = trial
    return trials


def load_recordings(task_dir: Path) -> dict[int, Path]:
    return {int(path.stem.removeprefix("trial")): path for path in sorted(task_dir.glob("record/**/trial*.npz"))}


def trace_stats(task_dir: Path) -> dict | None:
    records = []
    for path in sorted(task_dir.glob("trace/*.jsonl")):
        records += [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not records:
        return None
    stats = {"calls": len(records), "pred0_rms": float(np.mean([np.mean(r["pred0_rms_by_dim"]) for r in records]))}
    with_delta = [r for r in records if "delta_rms_by_dim" in r]
    if with_delta:
        by_step = np.array([r["delta_rms_by_step"] for r in with_delta])
        stats["pred1_rms"] = float(np.mean([np.mean(r["pred1_rms_by_dim"]) for r in with_delta]))
        stats["delta_rms"] = float(np.mean([np.mean(r["delta_rms_by_dim"]) for r in with_delta]))
        stats["delta_rms_by_dim"] = np.mean([r["delta_rms_by_dim"] for r in with_delta], axis=0).round(4).tolist()
        stats["delta_rms_first_10_steps"] = float(by_step[:, :10].mean())
        stats["delta_rms_later_steps"] = float(by_step[:, 10:].mean())
    return stats


def frame_size(path: Path | None) -> tuple[int, int]:
    if path is not None:
        with np.load(path) as data:
            frames = data["agentview"]
        if frames.ndim == 4 and len(frames):
            return int(frames.shape[1]), int(frames.shape[2])
    return 128, 128


def sheet_row(path: Path | None, size: tuple[int, int]) -> np.ndarray:
    height, width = size
    row = np.full((height, width * SHEET_FRAMES, 3), 255, dtype=np.uint8)
    if path is None:
        return row
    with np.load(path) as data:
        frames = data["agentview"]
    if len(frames) == 0:
        return row
    picks = np.linspace(0, len(frames) - 1, SHEET_FRAMES).round().astype(int)
    for column, index in enumerate(picks):
        frame = frames[index][:height, :width]
        row[: frame.shape[0], column * width : column * width + frame.shape[1]] = frame
    return row


def main(argv: list[str]) -> int:
    diag = Path(argv[1] if len(argv) > 1 else "outputs/libero/phase1_20261002/diag_20261003")
    task_names = sorted({path.name for run in RUNS for path in (diag / run).glob("*_task*") if path.is_dir()})
    summary: dict = {}
    lines = []
    for task_name in task_names:
        summary[task_name] = {}
        results = {run: load_results(diag / run / task_name) for run in RUNS}
        recordings = {run: load_recordings(diag / run / task_name) for run in RUNS}
        lines.append(f"== {task_name}")
        for run in RUNS:
            trials = results[run]
            wins = sorted(t for t, row in trials.items() if row.get("success"))
            win_steps = [trials[t]["policy_steps"] for t in wins]
            by_outcome = {"success": [], "failure": []}
            for trial, path in recordings[run].items():
                with np.load(path) as data:
                    stats = motion_stats(data["actions"], data["eef_pos"])
                outcome = "success" if trials.get(trial, {}).get("success") else "failure"
                by_outcome[outcome].append(stats)
            entry = {
                "successes": len(wins),
                "trials": len(trials),
                "success_median_steps": float(np.median(win_steps)) if win_steps else None,
                "motion": {
                    outcome: {
                        key: mean_of(rows, key)
                        for key in ("path_per_100_steps_m", "still_share", "reversal_share", "gripper_switches")
                    }
                    | {"trials": len(rows)}
                    for outcome, rows in by_outcome.items()
                },
                "trace": trace_stats(diag / run / task_name),
            }
            summary[task_name][run] = entry
            median = entry["success_median_steps"]
            lines.append(
                f"{run:9s} {len(wins):2d}/{len(trials):2d} median steps {'-' if median is None else int(median)}"
            )
            for outcome in ("success", "failure"):
                motion = entry["motion"][outcome]
                if motion["trials"]:
                    values = "  ".join(
                        f"{key} {'-' if motion[key] is None else round(motion[key], 3)}"
                        for key in ("path_per_100_steps_m", "still_share", "reversal_share", "gripper_switches")
                    )
                    lines.append(f"    {outcome:7s} n={motion['trials']:2d}  {values}")
            trace = entry["trace"]
            if trace:
                lines.append(
                    "    trace "
                    + "  ".join(
                        f"{key} {round(value, 4)}"
                        for key, value in trace.items()
                        if isinstance(value, float) or key == "calls"
                    )
                )
        failed_l2 = [t for t, row in results["l2_mip"].items() if not row.get("success")]
        picks = [t for t in sorted(failed_l2) if results["l1_fm2"].get(t, {}).get("success")][:SHEETS_PER_TASK]
        for trial in picks:
            paths = [recordings[run].get(trial) for run in RUNS]
            size = frame_size(next((path for path in paths if path is not None), None))
            sheet = np.concatenate([sheet_row(path, size) for path in paths], axis=0)
            write_png(diag / "sheets" / f"{task_name}_trial{trial:02d}.png", sheet)
        if picks:
            lines.append(f"    sheets (rows {', '.join(RUNS)}): trials {picks}")
    (diag / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
