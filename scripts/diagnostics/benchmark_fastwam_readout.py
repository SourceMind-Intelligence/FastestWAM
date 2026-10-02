"""Time the released Fast-WAM LIBERO action path under fixed-checkpoint access masks.

This is a synthetic-input latency probe, not a simulator or success evaluation.
"""

import json
import statistics
import time
from pathlib import Path

import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import OmegaConf


ROOT = Path("/root/evan/Fastest-WAM-evan/third_party/FastWAM-official")
CHECKPOINT = Path("/root/evan/Fastest-WAM-evan/assets/fastwam_official/libero_uncond_2cam224.pt")
OUT = Path("/root/evan/Fastest-WAM-evan/outputs/fastwam_readout/readout_latency_20260927.json")

OmegaConf.register_new_resolver("eval", eval, replace=True)
OmegaConf.register_new_resolver("max", lambda x: max(x), replace=True)
OmegaConf.register_new_resolver("split", lambda s, i: s.split("/")[int(i)], replace=True)
with initialize_config_dir(config_dir=str(ROOT / "configs"), version_base="1.3"):
    cfg = compose(config_name="sim_libero", overrides=["task=libero_uncond_2cam224_1e-4"])

model = instantiate(cfg.model, model_dtype=torch.bfloat16, device="cuda")
model.load_checkpoint(str(CHECKPOINT))
model = model.to("cuda").eval()
image = torch.zeros((1, 3, 224, 448), device="cuda", dtype=torch.bfloat16)
proprio = (
    torch.zeros((1, model.proprio_dim), device="cuda", dtype=torch.bfloat16)
    if model.proprio_dim is not None
    else None
)


def call(mode: str) -> float:
    if mode == "all":
        model.mot.video_readout_mode = "all"
        model.mot.video_readout_layer = None
    elif mode == "none":
        model.mot.video_readout_mode = "none"
        model.mot.video_readout_layer = None
    else:
        kind, layer = mode.split(":")
        model.mot.video_readout_mode = kind
        model.mot.video_readout_layer = int(layer)
    torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.inference_mode():
        result = model.infer_action(
            prompt="Put the object on the target.",
            input_image=image,
            action_horizon=32,
            proprio=proprio,
            num_inference_steps=10,
            sigma_shift=5.0,
            seed=42,
            compile_action_infer=False,
        )["action"]
    torch.cuda.synchronize()
    assert torch.isfinite(result).all()
    return (time.perf_counter() - start) * 1000.0


modes = ["all", "drop:27", "drop:28", "drop:29", "single:29", "none"]
for mode in modes:
    for _ in range(2):
        call(mode)

timings = {mode: [] for mode in modes}
for round_idx in range(10):
    order = modes if round_idx % 2 == 0 else list(reversed(modes))
    for mode in order:
        timings[mode].append(call(mode))

summary = {
    "source": "official Fast-WAM LIBERO checkpoint, fixed weights",
    "input": "synthetic zeros, two cameras concatenated to 224x448, zero proprio, fixed prompt",
    "action_horizon": 32,
    "denoising_steps": 10,
    "sigma_shift": 5.0,
    "compiled": False,
    "repetitions_per_mode": 10,
    "milliseconds": {
        mode: {
            "median": statistics.median(values),
            "minimum": min(values),
            "maximum": max(values),
            "values": values,
        }
        for mode, values in timings.items()
    },
}
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps({mode: round(summary["milliseconds"][mode]["median"], 2) for mode in modes}))
print(OUT)
