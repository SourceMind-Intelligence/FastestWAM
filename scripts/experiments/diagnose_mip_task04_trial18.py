#!/usr/bin/env python3
"""Replay MIP libero_10/task04 trial18 and dump On()/XY-distance each step."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import yaml


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8941)
    args = parser.parse_args()

    # LIBERO_CONFIG_PATH must already contain config.yaml (set by the shell wrapper).
    sys.path.insert(0, str(args.root / "benchmarks" / "libero"))
    from openwam2libero_interface import OpenWAMLiberoPolicy
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    suite_name, task_id, trial = "libero_10", 4, 18
    seed, max_steps, settle_steps = 42, 700, 30
    settle_action = np.zeros(7, np.float32)
    settle_action[-1] = -1.0

    task_suite = benchmark.get_benchmark_dict()[suite_name]()
    task = task_suite.get_task(task_id)
    init_states = task_suite.get_task_init_states(task_id)
    bddl = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=256, camera_widths=256)
    env.seed(seed)
    inner = env.env
    policy = OpenWAMLiberoPolicy(
        host="127.0.0.1",
        port=args.port,
        request_timeout=300,
        action_mode="eef",
        head_camera_key="agentview_image",
        left_wrist_camera_key="robot0_eye_in_hand_image",
        right_wrist_camera_key=None,
        image_transform="rotate_180",
        send_state=True,
        state_dim=10,
        action_dim=7,
    )

    def xy_dist(a: str, b: str):
        pa = inner.sim.data.body_xpos[inner.obj_body_id[a]]
        pb = inner.sim.data.body_xpos[inner.obj_body_id[b]]
        return float(np.linalg.norm(pa[:2] - pb[:2])), float(pa[2]), float(pb[2])

    def snapshot(step: int) -> dict:
        # BDDL: (On mug plate) -> plate.check_ontop(mug)
        p1 = inner.object_states_dict["plate_1"]
        p2 = inner.object_states_dict["plate_2"]
        mug_w = inner.object_states_dict["porcelain_mug_1"]
        mug_y = inner.object_states_dict["white_yellow_mug_1"]
        d_w, z_p1, z_w = xy_dist("plate_1", "porcelain_mug_1")
        d_y, z_p2, z_y = xy_dist("plate_2", "white_yellow_mug_1")
        return {
            "step": step,
            "success": bool(inner._check_success()),
            "on_white_left": bool(p1.check_ontop(mug_w)),
            "on_yellow_right": bool(p2.check_ontop(mug_y)),
            "xy_white_cm": round(d_w * 100.0, 2),
            "xy_yellow_cm": round(d_y * 100.0, 2),
            "z_ok_white": bool(z_p1 <= z_w),
            "z_ok_yellow": bool(z_p2 <= z_y),
            "contact_white": bool(p1.check_contact(mug_w)),
            "contact_yellow": bool(p2.check_contact(mug_y)),
        }

    obs = env.reset()
    obs = env.set_init_state(init_states[trial % len(init_states)])
    for _ in range(settle_steps):
        obs, _, _, _ = env.step(settle_action)
    policy.reset()

    log = []
    first_success = None
    for step in range(max_steps):
        obs, reward, done, _ = env.step(policy.act(obs, task.language))
        snap = snapshot(step + 1)
        snap["reward"] = float(reward)
        log.append(snap)
        if (step + 1) in {1, 50, 100, 175, 250, 350, 450, 500, 520, 525, 550, 600, 650, 700} or (
            step + 1
        ) % 100 == 0:
            print(json.dumps(snap), flush=True)
        if snap["success"] and first_success is None:
            first_success = snap
            break

    # Prefer dual-On; else minimize the worse of the two XY distances among dual-contact frames.
    both = [x for x in log if x["on_white_left"] and x["on_yellow_right"]]
    dual_contact = [x for x in log if x["contact_white"] and x["contact_yellow"]]
    best = None
    if both:
        best = min(both, key=lambda x: max(x["xy_white_cm"], x["xy_yellow_cm"]))
    elif dual_contact:
        best = min(dual_contact, key=lambda x: max(x["xy_white_cm"], x["xy_yellow_cm"]))
    else:
        best = min(log, key=lambda x: max(x["xy_white_cm"], x["xy_yellow_cm"]))

    around_525 = [x for x in log if 515 <= x["step"] <= 535]
    out = {
        "instruction": task.language,
        "trial": trial,
        "steps_run": len(log),
        "threshold_xy_cm": 3.0,
        "first_success": first_success,
        "final": log[-1],
        "best": best,
        "around_525": around_525,
        "white_on_frames": sum(1 for x in log if x["on_white_left"]),
        "yellow_on_frames": sum(1 for x in log if x["on_yellow_right"]),
        "both_on_frames": sum(1 for x in log if x["on_white_left"] and x["on_yellow_right"]),
        "note": "On(plate,mug) = contact AND plate_z<=mug_z AND xy_dist < 0.03m",
    }
    (args.out / "predicate_trace_summary.json").write_text(json.dumps(out, indent=2))
    (args.out / "predicate_trace.jsonl").write_text("\n".join(json.dumps(x) for x in log))
    print("==== SUMMARY ====", flush=True)
    print(json.dumps(out, indent=2), flush=True)
    env.close()
    policy.close()
    return 0


if __name__ == "__main__":
    # Fix accidental space in helper name if present after write — keep clean.
    raise SystemExit(main())
