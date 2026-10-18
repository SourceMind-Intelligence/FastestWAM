#!/usr/bin/env python3
"""Oracle per-window MSE lower bound for temporally constant action chunks.

Uses normalized training-reader targets and its padding/unification mask.
An oracle gets the full future chunk; a policy does not, so this is a lower bound.
"""
import argparse
import json
from pathlib import Path
import random
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
import torch
from hydra import compose, initialize_config_dir
from openwam.dataloader.registry import build_dataset


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--samples',type=int,default=1024)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();torch.set_num_threads(2)
    torch.manual_seed(42);np.random.seed(42);random.seed(42)
    root=Path(__file__).resolve().parents[2]
    with initialize_config_dir(config_dir=str(root/'configs'),version_base=None):
        cfg=compose(config_name='train',overrides=['dataloader=libero'])
    dataset=build_dataset(cfg.dataloader,split='train')
    indices=np.random.default_rng(42).choice(len(dataset),size=min(args.samples,len(dataset)),replace=False).tolist()
    subset=torch.utils.data.Subset(dataset,indices)
    loader=torch.utils.data.DataLoader(subset,batch_size=8,num_workers=0,collate_fn=list)
    values=[]
    for batch in loader:
        for sample in batch:
            a=sample['action'].float();valid=sample['action_mask'].bool()
            if valid.shape!=a.shape:
                raise ValueError(f'Unexpected mask shape: {valid.shape} vs {a.shape}')
            w=valid.float()
            mean=(a*w).sum(0)/w.sum(0).clamp_min(1)
            mse=(((a-mean)**2)*w).sum()/w.sum().clamp_min(1)
            values.append(mse.item())
    result=dict(seed=42,samples=len(values),dataset_windows=len(dataset),
        normalize_mode=str(cfg.dataloader.normalize_mode),horizon=int(cfg.dataloader.num_frames)-1,
        mean_oracle_constant_mse=float(np.mean(values)),
        quantiles={str(q):float(np.quantile(values,q)) for q in [0,.25,.5,.75,.9,1]},
        indices=indices,per_window_lower_bounds=values,
        caveat='Training data diagnostic, not held-out policy loss; sampled windows are correlated within demonstrations.')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print({k:v for k,v in result.items() if k not in ['indices','per_window_lower_bounds']})

if __name__=='__main__':main()
