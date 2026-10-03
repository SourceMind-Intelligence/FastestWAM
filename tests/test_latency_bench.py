"""The inference latency bench (scripts/diagnostics/latency_bench_20261003.py): modes, statistics,
stage timers on the DoT inference path, the joint-vs-video probe, client payloads and the summary."""

from __future__ import annotations

import base64
import importlib.util
import io
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

from tests.test_dot_architecture import ACTION_DIM, _attach_inference_inputs, _generate, _make_dot_arch

_PATH = Path(__file__).resolve().parents[1] / "scripts" / "diagnostics" / "latency_bench_20261003.py"
_SPEC = importlib.util.spec_from_file_location("latency_bench_20261003", _PATH)
bench = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(bench)


def _no_sync():
    return None


def _decode(b64: str) -> Image.Image:
    return Image.open(io.BytesIO(base64.b64decode(b64)))


def test_modes_parse_steps_cache_and_passes():
    assert bench.parse_mode("fm10-nocache") == {
        "mode": "fm10-nocache",
        "objective": "flow",
        "steps": 10,
        "cache": False,
        "passes": None,
    }
    assert bench.parse_mode("fm2")["cache"] is True
    assert bench.parse_mode("mip1") == {"mode": "mip1", "objective": "mip", "steps": 2, "cache": True, "passes": 1}
    for bad in ("fm0", "mip3", "fm", "flow2"):
        with pytest.raises(ValueError):
            bench.parse_mode(bad)


def test_default_modes_follow_the_objective_and_head():
    assert bench.default_modes("flow", "dot") == ["fm1", "fm2", "fm10"]
    assert bench.default_modes("flow", "joint_self_attn") == ["fm1", "fm2", "fm10", "fm10-nocache"]
    assert bench.default_modes("mip", "dot") == ["mip2", "mip1"]
    assert bench.default_modes("mip", "joint_self_attn") == ["mip2"]


def test_latency_summary_percentiles():
    out = bench.latency_summary([float(v) for v in range(1, 101)])
    assert out["n"] == 100
    assert out["p50"] == pytest.approx(50.5)
    assert out["p95"] == pytest.approx(95.05)
    assert out["min"] == 1.0 and out["max"] == 100.0
    assert bench.latency_summary([]) == {"n": 0}


def test_stage_timers_count_one_video_pass_and_a_head_pass_per_flow_step():
    torch.manual_seed(0)
    arch = _make_dot_arch()
    _attach_inference_inputs(arch)
    preprocess = arch.video_backbone.preprocess_input_for_inference
    timer = bench.instrument(arch, None, _no_sync)
    out = _generate(arch, [(1000.0, 1000.0), (600.0, 600.0), (300.0, 300.0), (0.0, 0.0)])
    assert out["actions"].shape == (8, ACTION_DIM)
    assert dict(timer.call_counts) == {"prep": 1, "video_backbone": 1, "action_head": 3}
    assert set(timer.call_seconds) == {"prep", "video_backbone", "action_head"}

    timer.restore()
    assert "_encode" not in vars(arch) and "_predict_actions" not in vars(arch)
    assert arch.video_backbone.preprocess_input_for_inference == preprocess
    timer.start_call()
    _generate(arch, [(1000.0, 1000.0), (0.0, 0.0)])
    assert not timer.call_counts


@pytest.mark.parametrize("passes", [1, 2])
def test_stage_timers_see_the_mip_head_passes(monkeypatch, passes):
    monkeypatch.setenv(bench.MIP_PASSES_ENV, str(passes))
    torch.manual_seed(0)
    arch = _make_dot_arch(action_objective="mip")
    _attach_inference_inputs(arch)
    timer = bench.instrument(arch, None, _no_sync)
    _generate(arch, [(1000.0, 1000.0), (0.0, 0.0)])
    timer.restore()
    assert timer.call_counts["action_head"] == passes
    assert timer.call_counts["video_backbone"] == 1


class _JointArch(torch.nn.Module):
    """Stands in for a joint video+action architecture: records how forward was called."""

    def __init__(self):
        super().__init__()
        self.video_backbone = SimpleNamespace(preprocess_input_for_inference=lambda **kw: {})
        self.calls = []

    def forward(self, noisy_actions, action_timestep, **inputs):
        self.calls.append({"video_only": noisy_actions is None, "inputs": sorted(inputs)})
        return torch.zeros(1), None if noisy_actions is None else torch.zeros(1)


class _Server:
    def __init__(self, arch):
        self.engine = SimpleNamespace(architecture=arch, _dit_cache=None)
        self.cfg = None
        self.resets = 0

    def reset(self):
        self.resets += 1

    def predict(self, payload):
        arch = self.engine.architecture
        for step in range(2):
            arch.forward(torch.zeros(1, 4, 3), torch.tensor([float(step)]), latents=torch.zeros(1), timestep=1)
        return {"action": [0.0]}


def _fake_torch():
    return SimpleNamespace(
        cuda=SimpleNamespace(synchronize=_no_sync),
        no_grad=torch.no_grad,
        compiler=SimpleNamespace(cudagraph_mark_step_begin=_no_sync),
    )


def test_probe_replays_the_first_joint_forward_with_and_without_actions():
    arch = _JointArch()
    server = _Server(arch)
    probe = bench.Bench(server, [{}], _fake_torch()).probe_joint_vs_video(warmup=1, reps=3)
    assert probe["joint_forward_ms"]["n"] == 3 and probe["video_only_forward_ms"]["n"] == 3
    assert "forward" not in vars(arch)
    replays = arch.calls[2:]
    assert [c["video_only"] for c in replays] == [False] * 4 + [True] * 4
    assert all(c["inputs"] == ["latents", "timestep"] for c in replays)


def test_probe_skips_the_dot_head():
    arch = _make_dot_arch()
    assert bench.Bench(_Server(arch), [{}], _fake_torch()).probe_joint_vs_video(warmup=1, reps=1) is None


def test_libero_payloads_match_the_client_slots_and_differ_between_turns():
    payloads = bench.build_payloads("libero", [0.5] * 10)
    assert len(payloads) == 4
    first = payloads[0]
    assert first["prompt"] == bench.BENCHMARKS["libero"]["prompt"]
    assert first["state"] == [0.5] * 10
    assert _decode(first["images"]["head_camera"]).size == (320, 256)
    assert _decode(first["images"]["left_wrist_camera"]).size == (160, 128)
    assert first["images"]["right_wrist_camera"] is None
    assert payloads[1]["images"]["head_camera"] != first["images"]["head_camera"]
    assert bench.build_payloads("libero", [0.5] * 10)[0] == first


def test_robodojo_payloads_send_three_native_views():
    payload = bench.build_payloads("robodojo", [0.0] * 20, count=1)[0]
    for slot in ("head_camera", "left_wrist_camera", "right_wrist_camera"):
        assert _decode(payload["images"][slot]).size == (640, 480)


def test_raw_state_takes_the_state_mean_when_its_width_matches():
    inner = SimpleNamespace(normalize_stats={"mean": np.arange(4, dtype=np.float32)})
    unified = SimpleNamespace(_inner=inner, _state_dst_index=np.array([0, 1, 2, 3]))
    assert bench.raw_state(SimpleNamespace(normalizer=unified), 10) == ([0.0, 1.0, 2.0, 3.0], "checkpoint state mean")
    assert bench.raw_state(SimpleNamespace(normalizer=None), 3) == ([0.0, 0.0, 0.0], "zeros")


def test_summary_lists_every_mode(tmp_path):
    report = {
        "label": "L1",
        "architecture": {"variant": "dot", "action_layers": 1, "objective": "flow"},
        "env": {"gpu": "NVIDIA H100 80GB HBM3", "driver": "570", "torch": "2.6", "cuda": "12.4", "compile": True},
        "modes": [
            {
                "mode": "fm2",
                "latency_ms": {"p50": 41.25, "p95": 44.0, "mean": 41.5},
                "peak_allocated_gib": 30.5,
                "breakdown": {
                    "stages_ms": {"prep": 9.0, "video_backbone": 25.0, "action_head": 3.0},
                    "per_chunk_counts": {"action_head": [2]},
                    "action_head_pass_ms": 1.5,
                },
            },
            {"mode": "mip2", "skipped": "checkpoint objective is flow"},
        ],
    }
    (tmp_path / "L1.json").write_text(json.dumps(report))
    table = bench.summarize([tmp_path / "L1.json"])
    rows = [line for line in table.splitlines() if line.startswith("| L1")]
    assert len(rows) == 2
    assert "| fm2 | 41.2 | 44.0 |" in rows[0] or "| fm2 | 41.3 | 44.0 |" in rows[0]
    assert "3.0 ([2] × 1.50)" in rows[0]
    assert "checkpoint objective is flow" in rows[1]
    assert "NVIDIA H100 80GB HBM3" in table


class _DotServer:
    """A policy server over the mock DoT architecture: each predict is one engine generate."""

    def __init__(self, arch, steps):
        from omegaconf import OmegaConf

        self.cfg = OmegaConf.create({"inference": {"denoise_steps": 10}})
        schedule = [(1000.0 * (1 - k / steps), 1000.0 * (1 - k / steps)) for k in range(steps + 1)]
        self.engine = SimpleNamespace(architecture=arch, _dit_cache=None)
        self.engine.generate = lambda conditions: _generate(arch, schedule)

    def reset(self):
        return None

    def predict(self, payload):
        return {"action": self.engine.generate({})["actions"][0].tolist()}


def _fake_cuda_torch():
    fake = _fake_torch()
    fake.cuda.empty_cache = _no_sync
    fake.cuda.reset_peak_memory_stats = _no_sync
    fake.cuda.mem_get_info = lambda: (60 * 2**30, 80 * 2**30)
    fake.cuda.max_memory_allocated = lambda: 30 * 2**30
    fake.cuda.max_memory_reserved = lambda: 32 * 2**30
    return fake


def test_a_mode_run_reports_latency_memory_and_the_stage_breakdown(monkeypatch):
    monkeypatch.delenv(bench.MIP_PASSES_ENV, raising=False)
    torch.manual_seed(0)
    arch = _make_dot_arch()
    _attach_inference_inputs(arch)
    server = _DotServer(arch, steps=2)
    result = bench.Bench(server, [{}], _fake_cuda_torch()).run_mode(bench.parse_mode("fm2"), 1, 4, 3)
    assert server.cfg.inference.denoise_steps == 2
    assert result["latency_ms"]["n"] == 4 and len(result["warmup_ms"]) == 1
    assert result["peak_allocated_gib"] == 30.0 and result["device_used_gib"] == 20.0
    breakdown = result["breakdown"]
    assert breakdown["n"] == 3
    assert breakdown["per_chunk_counts"] == {"action_head": [2], "generate": [1], "prep": [1], "video_backbone": [1]}
    assert breakdown["actions_shape"] == [8, ACTION_DIM] and breakdown["actions_finite"]
    assert breakdown["action_head_pass_ms"] > 0
    assert "_encode" not in vars(arch) and "generate" in vars(server.engine)
