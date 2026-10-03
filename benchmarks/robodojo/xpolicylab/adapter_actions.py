"""Run an XPolicyLab OpenWAM-family adapter on synthetic RoboDojo episodes, without Isaac Sim.

Each step follows XPolicyLab's batch loop (policy/OpenWAM/deploy.py, eval_one_episode_batch):
update_obs_batch for every env, get_action_batch at each chunk start, the chunk consumed one action
per step. Observations are synthetic and fixed by (env, step): three 640x480 views of a smooth scene
whose content moves every step, a fixed pose and the panel prompt. The output compares two adapters
(or two openwam trees) on identical inputs and times them; it says nothing about task success.

Run it in the policy server's environment with the PYTHONPATH the server gets (the run root that
holds XPolicyLab, then the openwam root). It imports nothing from openwam itself, so the adapter
picks the tree:

  PYTHONPATH=<run-root>:<openwam-root> python adapter_actions.py \
      --module XPolicyLab.policy.OpenWAM.model --ckpt-dir CK --openwam-root SRC --out ours.npz
  python adapter_actions.py --compare stock.npz ours.npz
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import sys
import time
from pathlib import Path

import numpy as np

CAMERAS = ("cam_head", "cam_left_wrist", "cam_right_wrist")
H, W = 480, 640
# Gripper open, each hand a little in front of and above its arm base (env-relative world frame,
# the frame XPolicyLab observations use); the bases sit at x = -0.3 / 0.3, y = -0.45, z = 0.765.
LEFT_POSE = np.array([-0.3, -0.2, 0.95, 0.7071068, 0.0, 0.0, 0.7071068])
RIGHT_POSE = np.array([0.3, -0.2, 0.95, 0.7071068, 0.0, 0.0, 0.7071068])
STATE_KEYS = ("left_ee_pose", "left_ee_joint_state", "right_ee_pose", "right_ee_joint_state")


def frame(env: int, step: int, camera: int, _grid=np.mgrid[0:H, 0:W].astype(np.float32)) -> np.ndarray:
    """A smooth (H, W, 3) uint8 view: a fixed gradient per (env, camera) and a disk that moves each step."""
    yy, xx = _grid
    phase = 0.7 * env + 1.3 * camera
    cy = H / 2 + 0.3 * H * np.sin(2 * np.pi * step / 90 + phase)
    cx = W / 2 + 0.35 * W * np.cos(2 * np.pi * step / 120 + phase)
    disk = ((yy - cy) ** 2 + (xx - cx) ** 2) < (0.12 * H) ** 2
    img = np.empty((H, W, 3), np.float32)
    img[..., 0] = 60 + 120 * xx / W + 20 * camera
    img[..., 1] = 50 + 100 * yy / H + 15 * env
    img[..., 2] = 90 + 40 * np.sin(xx / 37 + phase)
    img[disk] = (230, 200 - 10 * camera, 40 + 12 * env)
    return np.clip(img, 0, 255).astype(np.uint8)


def observation(env: int, step: int, prompt: str) -> dict:
    return {
        "vision": {cam: {"color": frame(env, step, k)} for k, cam in enumerate(CAMERAS)},
        "state": {
            "left_ee_pose": LEFT_POSE.copy(),
            "left_ee_joint_state": np.array([1.0]),
            "right_ee_pose": RIGHT_POSE.copy(),
            "right_ee_joint_state": np.array([1.0]),
        },
        "instruction": prompt,
        "env_idx": env,
    }


def as_array(chunk: list) -> np.ndarray:
    """One env's chunk of XPolicyLab ee dicts -> (T, 16): left pose 7, left grip 1, right pose 7, right grip 1."""
    return np.stack([np.concatenate([np.asarray(a[k], np.float64).reshape(-1) for k in STATE_KEYS]) for a in chunk])


def sha256(path: str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(args) -> None:
    module = importlib.import_module(args.module)
    model_cfg = {
        "policy_name": "OpenWAM",
        "bench_name": "RoboDojo",
        "task_name": "synthetic",
        "action_type": "ee",
        "env_cfg_type": args.env_cfg_type,
        "openwam_root": args.openwam_root,
        "ckpt_dir": args.ckpt_dir,
        "openwam_deploy_config": args.deploy_config,
        "device": args.device,
        "replan_steps": None,
        "default_instruction": "follow the instruction",
        "allow_dummy_policy": False,
        "seed": 42,
    }
    t0 = time.perf_counter()
    model = module.Model(model_cfg)
    load_s = time.perf_counter() - t0

    envs = list(range(args.envs))
    model.reset()
    chunks, call_steps, action_s, update_s = [], [], [], []
    step = 0
    while step < args.steps:
        t = time.perf_counter()
        model.update_obs_batch([observation(e, step, args.prompt) for e in envs])
        update_s.append(time.perf_counter() - t)
        t = time.perf_counter()
        actions = model.get_action_batch(envs)
        action_s.append(time.perf_counter() - t)
        chunks.append(np.stack([as_array(a) for a in actions]))
        call_steps.append(step)
        size = len(actions[0])
        for k in range(size):
            step += 1
            if step >= args.steps or k + 1 == size:
                break
            t = time.perf_counter()
            model.update_obs_batch([observation(e, step, args.prompt) for e in envs])
            update_s.append(time.perf_counter() - t)
        print(f"step {call_steps[-1]:4d}: chunk {chunks[-1].shape} in {action_s[-1]:.2f} s", flush=True)

    meta = {
        "module": args.module,
        "module_file": module.__file__,
        "module_sha256": sha256(module.__file__),
        "base_file": getattr(sys.modules.get(type(model).__mro__[1].__module__), "__file__", None),
        "ckpt_dir": args.ckpt_dir,
        "openwam_root": args.openwam_root,
        "openwam_file": getattr(sys.modules.get("openwam"), "__file__", None),
        "envs": args.envs,
        "steps": args.steps,
        "prompt": args.prompt,
        "load_s": round(load_s, 1),
        "argv": sys.argv,
    }
    try:
        import torch

        if torch.cuda.is_available():
            meta.update(
                gpu=torch.cuda.get_device_name(0),
                torch=torch.__version__,
                cuda=torch.version.cuda,
                peak_allocated_gib=round(torch.cuda.max_memory_allocated() / 2**30, 2),
            )
    except ImportError:
        pass
    np.savez(
        args.out,
        actions=np.stack(chunks),
        call_steps=np.array(call_steps),
        action_s=np.array(action_s),
        update_s=np.array(update_s),
        meta=json.dumps(meta),
    )
    timed = action_s[1:] or action_s
    print(
        json.dumps(
            {
                **{k: meta[k] for k in ("module_file", "openwam_file", "envs", "steps", "load_s")},
                "chunks": len(chunks),
                "chunk_s_median_after_first": round(float(np.median(timed)), 3),
                "update_s_median": round(float(np.median(update_s)), 4),
                "peak_allocated_gib": meta.get("peak_allocated_gib"),
            },
            indent=1,
        )
    )


def quat_angle_deg(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    dot = np.abs(np.sum(a * b, axis=-1)) / (np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1))
    return np.degrees(2 * np.arccos(np.clip(dot, -1.0, 1.0)))


def compare(path_a: str, path_b: str) -> dict:
    """Largest differences between two runs on the envs and chunks both have."""
    a, b = np.load(path_a), np.load(path_b)
    xa, xb = a["actions"], b["actions"]
    if not np.array_equal(a["call_steps"][: len(xb)], b["call_steps"][: len(xa)]):
        raise ValueError("the two runs did not request chunks at the same steps")
    n = min(len(xa), len(xb))
    envs = min(xa.shape[1], xb.shape[1])
    xa, xb = xa[:n, :envs], xb[:n, :envs]
    if xa.shape != xb.shape:
        raise ValueError(f"shapes differ: {xa.shape} vs {xb.shape}")
    out = {"chunks": n, "envs": envs, "steps_per_chunk": xa.shape[2]}
    for arm, (p, q, g) in {"left": (slice(0, 3), slice(3, 7), 7), "right": (slice(8, 11), slice(11, 15), 15)}.items():
        out[f"{arm}_position_max_mm"] = round(float(np.abs(xa[..., p] - xb[..., p]).max() * 1000), 4)
        out[f"{arm}_rotation_max_deg"] = round(float(quat_angle_deg(xa[..., q], xb[..., q]).max()), 4)
        out[f"{arm}_gripper_max"] = round(float(np.abs(xa[..., g] - xb[..., g]).max()), 5)
    out["identical"] = bool(np.array_equal(xa, xb))
    return out


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--module", default="XPolicyLab.policy.OpenWAM.model")
    parser.add_argument("--ckpt-dir")
    parser.add_argument("--openwam-root")
    parser.add_argument("--deploy-config", default=None)
    parser.add_argument("--env-cfg-type", default="arx_x5")
    parser.add_argument("--envs", type=int, default=10)
    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument("--prompt", default="Stack the bowls.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out")
    parser.add_argument("--compare", nargs=2, metavar=("A", "B"))
    args = parser.parse_args(argv)
    if args.compare:
        print(json.dumps(compare(*args.compare), indent=1))
        return
    if not (args.ckpt_dir and args.openwam_root and args.out):
        parser.error("--ckpt-dir, --openwam-root and --out are required unless --compare is given")
    run(args)


if __name__ == "__main__":
    main()
