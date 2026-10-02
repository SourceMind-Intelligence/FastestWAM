"""Model-path smoke test using the public RoboTwin release, without simulator claims."""
import time
from pathlib import Path

import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import OmegaConf

ROOT = Path('/root/evan/Fastest-WAM-evan/third_party/FastWAM-official')
CHECKPOINT = Path('/workspace/FastWAM/checkpoints/fastwam_release/robotwin_uncond_3cam_384.pt')
OmegaConf.register_new_resolver('eval', eval, replace=True)
OmegaConf.register_new_resolver('max', lambda x: max(x), replace=True)
OmegaConf.register_new_resolver('split', lambda s, i: s.split('/')[int(i)], replace=True)
with initialize_config_dir(config_dir=str(ROOT / 'configs'), version_base='1.3'):
    cfg = compose(config_name='sim_robotwin', overrides=['task=robotwin_uncond_3cam_384_1e-4'])
print('constructing model', flush=True)
model = instantiate(cfg.model, model_dtype=torch.bfloat16, device='cuda')
print('loading', CHECKPOINT, flush=True)
model.load_checkpoint(str(CHECKPOINT))
model = model.to('cuda').eval()
print('layers', model.mot.num_layers, flush=True)
image = torch.zeros((1, 3, 384, 384), device='cuda', dtype=torch.bfloat16)
for mode in ('all', 'single:15'):
    model.mot.video_readout_mode = 'all' if mode == 'all' else 'single'
    model.mot.video_readout_layer = None if mode == 'all' else 15
    start = time.monotonic()
    with torch.inference_mode():
        result = model.infer_action(
            prompt='Pick up the object.',
            input_image=image,
            action_horizon=24,
            num_inference_steps=1,
            seed=42,
            compile_action_infer=False,
        )['action']
    torch.cuda.synchronize()
    print(mode, 'shape', tuple(result.shape), 'finite', bool(torch.isfinite(result).all()),
          'seconds', round(time.monotonic()-start, 2), 'mean', round(float(result.mean()), 5), flush=True)
    if mode == 'all':
        baseline = result.clone()
    else:
        print('max_abs_difference', float((result-baseline).abs().max()), flush=True)
