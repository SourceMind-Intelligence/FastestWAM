#!/usr/bin/env python3
"""Single-GPU policy.compute benchmark with synthetic fixed camera observations.

Warm prompt cache, eager bf16, DiT cache disabled, no video decode. Times full
server.predict including preprocessing/normalization, excluding WebSocket and
simulator. Horizon=1 forces a newly generated action chunk on every request.
Run only after other CUDA work has left the selected GPU.
"""
import argparse,json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
import torch
from PIL import Image
from omegaconf import OmegaConf
from openwam.deploy.server import build_server_from_config


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--steps',type=int,nargs='+',default=[1,2,4])
    p.add_argument('--warmup',type=int,default=5)
    p.add_argument('--requests',type=int,default=30)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();torch.set_num_threads(4)
    cfg=OmegaConf.load('configs/deploy.yaml')
    cfg.optimization.compile.enabled=False;cfg.optimization.dit_cache.enabled=False
    cfg.inference.inference_horizon=1;cfg.inference.denoise_mode='sync'
    server=build_server_from_config(cfg,str(args.checkpoint.parent),device='cuda:0',ckpt_name=args.checkpoint.name)
    rng=np.random.default_rng(42)
    obs={'images':{k:Image.fromarray(rng.integers(0,256,(256,256,3),dtype=np.uint8))
                   for k in ['head_camera','left_wrist_camera']},
         'prompt':'pick up the black bowl and place it on the plate',
         'state':[0,0,0,1,0,0,0,1,0,0]}
    rows=[]
    try:
        for steps in args.steps:
            server.cfg.inference.denoise_steps=steps
            server.engine.cfg.inference.denoise_steps=steps
            server.reset()
            for _ in range(args.warmup):server.predict(obs)
            torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
            times=[]
            for _ in range(args.requests):
                torch.cuda.synchronize();start=time.perf_counter()
                response=server.predict(obs)
                torch.cuda.synchronize();times.append((time.perf_counter()-start)*1000)
                if not np.isfinite(response['action']).all():raise ValueError('Nonfinite action')
            rows.append(dict(denoise_steps=steps,p50_ms=float(np.median(times)),
                p95_ms=float(np.quantile(times,.95)),mean_ms=float(np.mean(times)),
                peak_allocated_bytes=torch.cuda.max_memory_allocated(),latencies_ms=times))
            args.output.parent.mkdir(parents=True,exist_ok=True)
            args.output.write_text(json.dumps(dict(checkpoint=str(args.checkpoint),
                gpu=torch.cuda.get_device_name(),torch_version=torch.__version__,warmup=args.warmup,
                requests=args.requests,compile=False,dit_cache=False,prompt_cache=True,
                action_horizon=32,execution_horizon=1,synthetic_fixed_observations=True,
                timing='server.predict with CUDA synchronize; excludes websocket/simulator',results=rows),indent=2)+'\n')
    finally:server.shutdown()

if __name__=='__main__':main()
