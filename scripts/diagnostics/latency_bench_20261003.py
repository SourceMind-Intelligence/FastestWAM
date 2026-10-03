#!/usr/bin/env python3
"""Inference latency and GPU memory of a WAM checkpoint, measured the way the eval clients are served.

The project is named for speed but had never measured a single inference latency (2026-10-03). This
times one action chunk: the policy server's ``predict()`` on an empty action buffer, which decodes the
PNG payload, composes the camera canvas, normalizes the state and runs the engine's ``generate()``.
That is the latency an eval client waits for at each replan. Network transfer is not included.

The model is loaded once with ``configs/deploy.yaml`` (torch.compile and the DiT velocity cache on, as
in the LIBERO and RoboDojo evals), then each mode is timed in turn:

    fmN          flow matching with N denoising steps (1, 2, 10, ...)
    fmN-nocache  the same with the DiT velocity cache off, which only the joint (deep-head) loop uses
    mip2         the MIP objective's two passes from zeros
    mip1         DoT only: the first MIP pass alone (OPENWAM_DOT_MIP_PASSES=1)

Per mode: warm-up calls, then ``--iters`` timed calls (p50/p90/p95/p99, mean), peak allocated and
reserved GPU memory, and a separate breakdown pass with synchronized timers around the stages:
preprocess (text embedding, VAE encode of the observed frame), the video backbone pass and action-head
passes (DoT), or the joint video+action forwards (deep head), with how many of each ran per chunk.
The timed calls carry no extra synchronization; the breakdown pass does, so its total runs slightly higher.
For a joint (deep-head) checkpoint the probe also times one joint forward against a video-only forward
on the same inputs, an estimate of what the action expert adds to each step.

``--force-objective mip`` runs another objective's inference path on the same weights; the actions are
meaningless but the compute is the same, which gives a deep MIP latency from a flow checkpoint.

    python scripts/diagnostics/latency_bench_20261003.py --ckpt-dir <run dir> --label L1 \
        --benchmark libero --out outputs/latency_20261003/L1.json
    python scripts/diagnostics/latency_bench_20261003.py --summarize outputs/latency_20261003
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import math
import os
import re
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "third_party"))

logger = logging.getLogger("latency_bench")

MIP_PASSES_ENV = "OPENWAM_DOT_MIP_PASSES"
MODE_RE = re.compile(r"^(?:fm(?P<steps>\d+)(?P<nocache>-nocache)?|mip(?P<passes>[12]))$")

# What an eval client sends, per benchmark. LIBERO: the 256x256 agentview and wrist frames, resized to
# their L-shape slots by the client as benchmarks/libero does, a 10-D state and the raw task language.
# RoboDojo: three 640x480 views left for the server to compose, a 20-D (two-arm EEF) state.
BENCHMARKS = {
    "libero": {
        "cameras": {"head_camera": (256, 256), "left_wrist_camera": (256, 256)},
        "client_resize": True,
        "state_dim": 10,
        "prompt": "put both the alphabet soup and the tomato sauce in the basket",
    },
    "robodojo": {
        "cameras": {"head_camera": (480, 640), "left_wrist_camera": (480, 640), "right_wrist_camera": (480, 640)},
        "client_resize": False,
        "state_dim": 20,
        "prompt": "Stack the bowls.",
    },
}


# ---------------------------------------------------------------------------
# Statistics and modes
# ---------------------------------------------------------------------------


def latency_summary(samples_ms: list[float]) -> dict:
    """Percentiles of a latency sample, in milliseconds."""
    arr = np.asarray(samples_ms, dtype=np.float64)
    if arr.size == 0:
        return {"n": 0}
    return {
        "n": int(arr.size),
        "p50": round(float(np.percentile(arr, 50)), 3),
        "p90": round(float(np.percentile(arr, 90)), 3),
        "p95": round(float(np.percentile(arr, 95)), 3),
        "p99": round(float(np.percentile(arr, 99)), 3),
        "mean": round(float(arr.mean()), 3),
        "std": round(float(arr.std()), 3),
        "min": round(float(arr.min()), 3),
        "max": round(float(arr.max()), 3),
    }


def parse_mode(mode: str) -> dict:
    """``fm10-nocache`` -> objective, denoise steps, cache flag and MIP passes."""
    match = MODE_RE.match(mode.strip())
    if match is None:
        raise ValueError(f"unknown mode {mode!r}; expected fmN, fmN-nocache, mip1 or mip2")
    if match.group("steps"):
        steps = int(match.group("steps"))
        if steps < 1:
            raise ValueError(f"mode {mode!r} needs at least one denoising step")
        return {"mode": mode, "objective": "flow", "steps": steps, "cache": not match.group("nocache"), "passes": None}
    return {"mode": mode, "objective": "mip", "steps": 2, "cache": True, "passes": int(match.group("passes"))}


def default_modes(objective: str, variant: str) -> list[str]:
    """The modes Evan's ask covers for this checkpoint's objective and head."""
    if objective == "mip":
        return ["mip2", "mip1"] if variant == "dot" else ["mip2"]
    modes = ["fm1", "fm2", "fm10"]
    if variant != "dot":
        modes.append("fm10-nocache")
    return modes


# ---------------------------------------------------------------------------
# Stage timers
# ---------------------------------------------------------------------------


class StageTimer:
    """Synchronized wall-clock timers patched onto object attributes, removable afterwards.

    Each wrapped call synchronizes the device before and after, so a stage's time is its own even
    though CUDA launches are asynchronous. Stages are per-call totals, reset by :meth:`start_call`.
    """

    def __init__(self, sync: Callable[[], None]):
        self._sync = sync
        self._patched: list[tuple[Any, str, Any, bool]] = []
        self.call_seconds: dict[str, float] = defaultdict(float)
        self.call_counts: dict[str, int] = defaultdict(int)
        self.last_results: dict[str, Any] = {}

    def wrap(self, obj: Any, name: str, stage: str) -> None:
        original = getattr(obj, name)
        own = name in vars(obj)

        def timed(*args, **kwargs):
            self._sync()
            start = time.perf_counter()
            try:
                result = original(*args, **kwargs)
            finally:
                self._sync()
                self.call_seconds[stage] += time.perf_counter() - start
                self.call_counts[stage] += 1
            self.last_results[stage] = result
            return result

        object.__setattr__(obj, name, timed)
        self._patched.append((obj, name, original, own))

    def start_call(self) -> None:
        self.call_seconds = defaultdict(float)
        self.call_counts = defaultdict(int)
        self.last_results = {}

    def restore(self) -> None:
        for obj, name, original, own in reversed(self._patched):
            if own:
                object.__setattr__(obj, name, original)
            else:
                object.__delattr__(obj, name)
        self._patched.clear()


def instrument(arch: Any, engine: Any, sync: Callable[[], None]) -> StageTimer:
    """Wrap the inference stages of an architecture (and its engine) in synchronized timers.

    ``generate`` is the engine call, ``prep`` the video backbone's input preparation (cached text
    embedding, VAE encode of the observed frame), ``video_backbone``/``action_head`` the DoT path's one
    clean-prefix pass and its head passes, and ``joint_forward`` each video+action forward of the
    joint denoising loop.
    """
    timer = StageTimer(sync)
    if engine is not None:
        timer.wrap(engine, "generate", "generate")
    timer.wrap(arch.video_backbone, "preprocess_input_for_inference", "prep")
    if hasattr(arch, "_predict_actions") and hasattr(arch, "_encode"):
        timer.wrap(arch, "_encode", "video_backbone")
        timer.wrap(arch, "_predict_actions", "action_head")
    else:
        timer.wrap(arch, "forward", "joint_forward")
    return timer


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------


def synthetic_frame(height: int, width: int, seed: int) -> np.ndarray:
    """A deterministic RGB frame: a smooth color field with a few flat-colored boxes as objects."""
    from PIL import Image

    rng = np.random.default_rng(seed)
    coarse = rng.integers(0, 256, size=(6, 8, 3), dtype=np.uint8)
    frame = np.array(Image.fromarray(coarse).resize((width, height), Image.Resampling.BICUBIC))
    for _ in range(5):
        h = int(rng.integers(max(2, height // 10), max(3, height // 4)))
        w = int(rng.integers(max(2, width // 10), max(3, width // 4)))
        y = int(rng.integers(0, height - h))
        x = int(rng.integers(0, width - w))
        frame[y : y + h, x : x + w] = rng.integers(0, 256, size=3, dtype=np.uint8)
    return frame


def build_payloads(benchmark: str, state: list[float], count: int = 4, seed: int = 0) -> list[dict]:
    """``count`` client payloads with different frames, built with the eval clients' own helpers."""
    from benchmarks.utils.client import build_payload, encode_numpy_b64, resize_for_lshape_slot

    spec = BENCHMARKS[benchmark]
    payloads = []
    for k in range(count):
        images = {}
        for slot_index, (slot, (height, width)) in enumerate(spec["cameras"].items()):
            frame = synthetic_frame(height, width, seed=seed + 100 * k + slot_index)
            if spec["client_resize"]:
                frame = np.asarray(resize_for_lshape_slot(frame, slot))
            images[slot] = encode_numpy_b64(np.ascontiguousarray(frame))
        payloads.append(
            build_payload(
                images["head_camera"],
                images.get("left_wrist_camera"),
                images.get("right_wrist_camera"),
                prompt=spec["prompt"],
                state=state,
            )
        )
    return payloads


def raw_state(arch: Any, default_dim: int) -> tuple[list[float], str]:
    """A plausible raw robot state: the checkpoint's state mean when the normalizer exposes it."""
    normalizer = getattr(arch, "normalizer", None)
    dim = default_dim
    index = getattr(normalizer, "_state_dst_index", None)
    if index is not None:
        dim = len(index)
    inner = getattr(normalizer, "_inner", normalizer)
    inner = getattr(inner, "_proprio_inner", inner)
    for attr in ("normalize_stats", "stats"):
        stats = getattr(inner, attr, None)
        if isinstance(stats, dict) and stats.get("mean") is not None:
            mean = np.ravel(np.asarray(stats["mean"], dtype=np.float64))
            if mean.size == dim:
                return [float(v) for v in mean], "checkpoint state mean"
    return [0.0] * dim, "zeros"


# ---------------------------------------------------------------------------
# Measurement
# ---------------------------------------------------------------------------


def _gib(nbytes: float) -> float:
    return round(float(nbytes) / 2**30, 3)


def _driver_version() -> Optional[str]:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=30,
            check=True,
        ).stdout
        return out.strip().splitlines()[0].strip() if out.strip() else None
    except (OSError, subprocess.SubprocessError):
        return None


def _param_millions(module: Any) -> Optional[float]:
    if module is None or not hasattr(module, "parameters"):
        return None
    return round(sum(p.numel() for p in module.parameters()) / 1e6, 2)


def _select(cfg: Any, key: str, default: Any = None) -> Any:
    from omegaconf import OmegaConf

    return OmegaConf.select(cfg, key, default=default)


def describe_architecture(cfg: Any, arch: Any) -> dict:
    vb = getattr(arch, "video_backbone", None)
    ab = getattr(arch, "action_backbone", None)
    return {
        "class": type(arch).__name__,
        "variant": _select(cfg, "model.architecture.variant"),
        "objective": getattr(arch, "_action_objective", _select(cfg, "model.architecture.action_objective")),
        "video_layers": getattr(vb, "num_layers", None),
        "action_layers": getattr(ab, "num_layers", None),
        "action_dim": getattr(arch, "action_dim", None),
        "params_m": {
            "total": _param_millions(arch),
            "video_dit": _param_millions(getattr(vb, "dit", None)),
            "action_backbone": _param_millions(ab),
            "dot_fusion": _param_millions(getattr(arch, "dot_fusion", None)),
        },
        "video": {
            "frames": _select(cfg, "inference.video_num_frames"),
            "action_frames": _select(cfg, "inference.num_frames"),
            "height": _select(cfg, "inference.height"),
            "width": _select(cfg, "inference.width"),
            "history_num_frames": _select(cfg, "dataloader.history_num_frames", 0),
        },
    }


class Bench:
    """One loaded checkpoint, timed mode by mode."""

    def __init__(self, server: Any, payloads: list[dict], torch_mod: Any):
        self.server = server
        self.engine = server.engine
        self.arch = server.engine.architecture
        self.payloads = payloads
        self.torch = torch_mod
        self._cache = self.engine._dit_cache
        self._turn = 0

    def sync(self) -> None:
        self.torch.cuda.synchronize()

    def chunk(self) -> float:
        """One action chunk as a client waits for it, in milliseconds."""
        payload = self.payloads[self._turn % len(self.payloads)]
        self._turn += 1
        self.server.reset()
        self.sync()
        start = time.perf_counter()
        self.server.predict(payload)
        self.sync()
        return (time.perf_counter() - start) * 1e3

    def apply(self, spec: dict) -> None:
        from omegaconf import OmegaConf

        OmegaConf.update(self.server.cfg, "inference.denoise_steps", spec["steps"], merge=False)
        self.engine._dit_cache = self._cache if spec["cache"] else None
        if spec["passes"] is not None:
            os.environ[MIP_PASSES_ENV] = str(spec["passes"])
        else:
            os.environ.pop(MIP_PASSES_ENV, None)

    def run_mode(self, spec: dict, warmup: int, iters: int, breakdown_iters: int) -> dict:
        torch = self.torch
        self.apply(spec)
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        warm = [round(self.chunk(), 3) for _ in range(warmup)]
        samples = [self.chunk() for _ in range(iters)]
        free, total = torch.cuda.mem_get_info()
        result = {
            **spec,
            "dit_cache": self.engine._dit_cache is not None,
            "warmup_ms": warm,
            "latency_ms": latency_summary(samples),
            "samples_ms": [round(s, 3) for s in samples],
            "peak_allocated_gib": _gib(torch.cuda.max_memory_allocated()),
            "peak_reserved_gib": _gib(torch.cuda.max_memory_reserved()),
            "device_used_gib": _gib(total - free),
        }
        result["breakdown"] = self.breakdown(breakdown_iters)
        return result

    def breakdown(self, iters: int) -> dict:
        timer = instrument(self.arch, self.engine, self.sync)
        per_call = []
        try:
            for _ in range(iters):
                timer.start_call()
                total_ms = self.chunk()
                record = {
                    "total_ms": total_ms,
                    "stages_ms": {k: v * 1e3 for k, v in timer.call_seconds.items()},
                    "counts": dict(timer.call_counts),
                }
                if self.engine._dit_cache is not None:
                    stats = self.engine._dit_cache.stats
                    record["dit_cache"] = {k: stats.get(k) for k in ("total_steps", "total_skips") if k in stats}
                generated = timer.last_results.get("generate")
                actions = generated.get("actions") if isinstance(generated, dict) else None
                if actions is not None:
                    actions = np.asarray(actions)
                    record["actions_shape"] = list(actions.shape)
                    record["actions_finite"] = bool(np.isfinite(actions).all())
                per_call.append(record)
        finally:
            timer.restore()
        stages = sorted({k for r in per_call for k in r["stages_ms"]})
        counts = sorted({k for r in per_call for k in r["counts"]})
        out = {
            "n": len(per_call),
            "total_ms": round(float(np.mean([r["total_ms"] for r in per_call])), 3) if per_call else None,
            "stages_ms": {k: round(float(np.mean([r["stages_ms"].get(k, 0.0) for r in per_call])), 3) for k in stages},
            "per_chunk_counts": {k: sorted({r["counts"].get(k, 0) for r in per_call}) for k in counts},
            "actions_shape": per_call[-1].get("actions_shape") if per_call else None,
            "actions_finite": all(r.get("actions_finite", True) for r in per_call),
        }
        heads = [r for r in per_call if r["counts"].get("action_head")]
        if heads:
            out["action_head_pass_ms"] = round(
                float(np.mean([r["stages_ms"]["action_head"] / r["counts"]["action_head"] for r in heads])), 3
            )
        joints = [r for r in per_call if r["counts"].get("joint_forward")]
        if joints:
            out["joint_forward_ms"] = round(
                float(np.mean([r["stages_ms"]["joint_forward"] / r["counts"]["joint_forward"] for r in joints])), 3
            )
        caches = [r["dit_cache"] for r in per_call if r.get("dit_cache")]
        if caches:
            out["dit_cache"] = {
                key: sorted({c.get(key) for c in caches if c.get(key) is not None})
                for key in ("total_steps", "total_skips")
            }
        return out

    def probe_joint_vs_video(self, warmup: int, reps: int) -> Optional[dict]:
        """Time one joint forward against a video-only forward on the inputs of a real first step."""
        arch = self.arch
        if hasattr(arch, "_predict_actions"):
            return None
        captured: dict = {}
        original = arch.forward

        def capture(*args, **kwargs):
            if "args" not in captured:
                captured["args"], captured["kwargs"] = args, kwargs
            return original(*args, **kwargs)

        object.__setattr__(arch, "forward", capture)
        try:
            self.chunk()
        finally:
            object.__delattr__(arch, "forward")
        if "args" not in captured:
            return None
        args, kwargs = captured["args"], captured["kwargs"]
        torch = self.torch

        def timed(fn) -> float:
            samples = []
            for i in range(warmup + reps):
                self.sync()
                start = time.perf_counter()
                with torch.no_grad():
                    torch.compiler.cudagraph_mark_step_begin()
                    fn()
                self.sync()
                if i >= warmup:
                    samples.append((time.perf_counter() - start) * 1e3)
            return latency_summary(samples)

        joint = timed(lambda: original(*args, **kwargs))
        video_only = timed(lambda: original(None, None, **kwargs))
        return {"joint_forward_ms": joint, "video_only_forward_ms": video_only}


def load_server(args) -> tuple[Any, float]:
    from omegaconf import OmegaConf

    from openwam.deploy.server import (
        _apply_compile_enabled_override,
        _load_deploy_yaml,
        _validate_inference_config,
        build_server_from_config,
    )

    cfg = _load_deploy_yaml(args.config)
    if args.overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(args.overrides))
    _validate_inference_config(cfg)
    _apply_compile_enabled_override(cfg, args.compile)
    start = time.perf_counter()
    server = build_server_from_config(cfg=cfg, ckpt_dir=args.ckpt_dir, device=args.device, ckpt_name=args.ckpt_name)
    return server, time.perf_counter() - start


def run(args) -> dict:
    import torch

    from openwam.deploy.model_loader import _find_latest_checkpoint

    torch.cuda.set_device(torch.device(args.device))
    server, load_s = load_server(args)
    arch = server.engine.architecture
    if args.force_objective:
        arch._action_objective = args.force_objective
    info = describe_architecture(server.cfg, arch)
    objective = info["objective"]
    variant = str(info["variant"] or "")
    state, state_source = raw_state(arch, BENCHMARKS[args.benchmark]["state_dim"])
    payloads = build_payloads(args.benchmark, state)
    bench = Bench(server, payloads, torch)
    weights_gib = _gib(torch.cuda.memory_allocated())

    report: dict = {
        "label": args.label,
        "ckpt_dir": str(args.ckpt_dir),
        "ckpt_file": os.path.basename(
            os.path.join(args.ckpt_dir, args.ckpt_name) if args.ckpt_name else _find_latest_checkpoint(args.ckpt_dir)
        ),
        "benchmark": args.benchmark,
        "forced_objective": args.force_objective,
        "architecture": info,
        "env": {
            "gpu": torch.cuda.get_device_name(torch.device(args.device)),
            "driver": _driver_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "compile": bool(_select(server.cfg, "optimization.compile.enabled", False)),
            "dit_cache_config": {
                "enabled": bool(_select(server.cfg, "optimization.dit_cache.enabled", False)),
                "cosine_threshold": _select(server.cfg, "optimization.dit_cache.cosine_threshold"),
                "max_skips": _select(server.cfg, "optimization.dit_cache.max_skips"),
            },
            "inference_horizon": _select(server.cfg, "inference.inference_horizon"),
        },
        "inputs": {
            "frames": "synthetic, 4 payloads in turn",
            "cameras": {k: list(v) for k, v in BENCHMARKS[args.benchmark]["cameras"].items()},
            "client_resize": BENCHMARKS[args.benchmark]["client_resize"],
            "state_dim": len(state),
            "state": state_source,
            "prompt": BENCHMARKS[args.benchmark]["prompt"],
        },
        "load_s": round(load_s, 1),
        "weights_gib": weights_gib,
        "modes": [],
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    modes = args.modes.split(",") if args.modes else default_modes(objective, variant)
    for mode in modes:
        spec = parse_mode(mode)
        if spec["objective"] != objective:
            report["modes"].append({**spec, "skipped": f"checkpoint objective is {objective}"})
            continue
        if spec["passes"] == 1 and not hasattr(arch, "_mip_inference_passes"):
            report["modes"].append({**spec, "skipped": "this code has no single-pass MIP mode"})
            continue
        logger.info("[latency] %s %s: warm-up %d, timed %d", args.label, mode, args.warmup, args.iters)
        try:
            result = bench.run_mode(spec, args.warmup, args.iters, args.breakdown_iters)
        except Exception as exc:  # keep the other modes' numbers
            logger.exception("[latency] %s %s failed", args.label, mode)
            result = {**spec, "error": f"{type(exc).__name__}: {exc}"}
        report["modes"].append(result)
        lat = result.get("latency_ms", {})
        logger.info("[latency] %s %s: p50 %s ms, p95 %s ms", args.label, mode, lat.get("p50"), lat.get("p95"))
        _write(args.out, report)
    if args.probe and not hasattr(arch, "_predict_actions"):
        bench.apply({**parse_mode("fm1-nocache"), "steps": 1 if objective == "flow" else 2})
        try:
            report["probe"] = bench.probe_joint_vs_video(warmup=5, reps=args.probe)
        except Exception as exc:
            logger.exception("[latency] %s probe failed", args.label)
            report["probe"] = {"error": f"{type(exc).__name__}: {exc}"}
    report["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    _write(args.out, report)
    return report


def _write(path: Optional[str], report: dict) -> None:
    if not path:
        return
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".tmp")
    tmp.write_text(json.dumps(report, indent=1, ensure_ascii=False))
    tmp.replace(out)


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------


def _fmt(value: Any, digits: int = 1) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "-"
    if isinstance(value, (int, float)):
        return f"{value:.{digits}f}"
    return str(value)


def summarize(paths: list[Path]) -> str:
    """A Markdown table of every mode in the given reports."""
    header = (
        "| Model | Head | Mode | p50 ms | p95 ms | mean ms | Peak alloc GiB | Prep ms | Video ms | Head ms "
        "| Joint fwd ms × n | Cache skips | Note |"
    )
    lines = [header, "|" + "---|" * 13]
    env = None
    for path in paths:
        report = json.loads(path.read_text())
        env = env or report.get("env")
        info = report.get("architecture", {})
        head = f"{info.get('variant')} {info.get('action_layers')}-layer {info.get('objective')}"
        for mode in report.get("modes", []):
            note = mode.get("skipped") or mode.get("error") or ""
            if report.get("forced_objective"):
                note = (note + " " if note else "") + f"{report['forced_objective']} path on these weights"
            lat = mode.get("latency_ms", {})
            br = mode.get("breakdown", {})
            stages = br.get("stages_ms", {})
            joint = ""
            if br.get("joint_forward_ms") is not None:
                joint = f"{_fmt(br['joint_forward_ms'])} × {br.get('per_chunk_counts', {}).get('joint_forward')}"
            head_ms = stages.get("action_head")
            if br.get("action_head_pass_ms") is not None:
                head_ms = f"{_fmt(head_ms)} ({br.get('per_chunk_counts', {}).get('action_head')} × {_fmt(br['action_head_pass_ms'], 2)})"
            skips = br.get("dit_cache", {}).get("total_skips")
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(report.get("label")),
                        head,
                        str(mode.get("mode")),
                        _fmt(lat.get("p50")),
                        _fmt(lat.get("p95")),
                        _fmt(lat.get("mean")),
                        _fmt(mode.get("peak_allocated_gib"), 2),
                        _fmt(stages.get("prep")),
                        _fmt(stages.get("video_backbone")),
                        head_ms if isinstance(head_ms, str) else _fmt(head_ms),
                        joint or "-",
                        "-" if skips is None else str(skips),
                        note,
                    ]
                )
                + " |"
            )
        probe = report.get("probe")
        if probe and "joint_forward_ms" in probe:
            lines.append(
                f"| {report.get('label')} | probe | one forward | joint {_fmt(probe['joint_forward_ms'].get('p50'))} "
                f"| video only {_fmt(probe['video_only_forward_ms'].get('p50'))} | | | | | | | | p50 of each |"
            )
    if env:
        lines.append("")
        lines.append(
            f"GPU {env.get('gpu')}, driver {env.get('driver')}, torch {env.get('torch')} (CUDA {env.get('cuda')}), "
            f"compile {env.get('compile')}."
        )
    return "\n".join(lines) + "\n"


def _parse_bool(value: str) -> bool:
    lowered = value.strip().lower()
    if lowered in ("true", "1", "yes", "on"):
        return True
    if lowered in ("false", "0", "no", "off"):
        return False
    raise argparse.ArgumentTypeError(f"expected true or false, got {value!r}")


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--ckpt-dir", help="Checkpoint directory (config.yaml + checkpoint_step_*.safetensors)")
    parser.add_argument("--ckpt-name", default=None, help="Checkpoint file; default the latest")
    parser.add_argument("--label", default=None, help="Name for this checkpoint in the report")
    parser.add_argument("--benchmark", choices=sorted(BENCHMARKS), default="libero")
    parser.add_argument("--modes", default=None, help="Comma list, e.g. fm1,fm2,fm10,fm10-nocache,mip2,mip1")
    parser.add_argument("--force-objective", choices=["flow", "mip"], default=None)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--breakdown-iters", type=int, default=20)
    parser.add_argument("--probe", type=int, default=20, help="Joint vs video-only forward reps (0 = off)")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--config", default=None, help="Deploy yaml (default configs/deploy.yaml)")
    parser.add_argument("--compile", type=_parse_bool, default=None, help="Override optimization.compile.enabled")
    parser.add_argument("--out", default=None, help="Report JSON path")
    parser.add_argument("--summarize", default=None, help="Print a Markdown table of the reports in this dir")
    parser.add_argument("overrides", nargs="*", help="OmegaConf dotlist overrides of the deploy yaml")
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = build_argparser().parse_args(argv)
    if args.summarize:
        paths = sorted(p for p in Path(args.summarize).glob("*.json"))
        sys.stdout.write(summarize(paths))
        return 0
    if not args.ckpt_dir:
        raise SystemExit("--ckpt-dir is required")
    args.label = args.label or Path(args.ckpt_dir).name
    for mode in (args.modes or "").split(","):
        if mode:
            parse_mode(mode)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    report = run(args)
    failed = [m["mode"] for m in report["modes"] if m.get("error")]
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
