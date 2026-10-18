#!/usr/bin/env python3
"""Probe a trained action head with synthetic bridges, without loading VideoDiT.

This tests temporal expressivity, NOT policy success or real visual conditioning.
Geometry defaults match this repository's Wan2.2-5B DoT checkpoints.
"""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import torch
from safetensors import safe_open
from openwam.model.action_backbone.separate_action_dit import ActionDiT
from openwam.model.architectures.utils.common import evenly_spaced_bridge_layers


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--num-heads',type=int,default=24)
    p.add_argument('--video-layers',type=int,default=30)
    p.add_argument('--dtype', choices=['float32', 'bfloat16'], default='float32')
    p.add_argument('--device', default='cpu')
    args=p.parse_args();torch.set_num_threads(4);torch.manual_seed(42)
    with safe_open(str(args.checkpoint),framework='pt',device='cpu') as f:
        state={k.removeprefix('action_backbone.'):f.get_tensor(k)
               for k in f.keys() if k.startswith('action_backbone.')}
    dim,action_dim=state['action_encoder.weight'].shape
    layers=sum(k.endswith('.self_attn.q.weight') for k in state)
    kv_dim=state['video_projs.0.weight'].shape[1]
    q_width=state['blocks.0.self_attn.q.weight'].shape[0]
    model=ActionDiT(action_dim=action_dim,dim=dim,ffn_dim=dim*4,
        num_heads=args.num_heads,attn_head_dim=q_width//args.num_heads,num_layers=layers,
        video_dim=kv_dim,bridge_layers=evenly_spaced_bridge_layers(args.video_layers,layers),
        text_dim=state['text_embedding.0.weight'].shape[1]).eval()
    model.load_state_dict(state,strict=True);del state
    dtype=getattr(torch,args.dtype)
    model=model.to(device=args.device,dtype=dtype)
    bridges={i:torch.randn(1,64,kv_dim,device=args.device,dtype=dtype) for i in model.bridge_layers}
    context=torch.randn(1,8,model.text_dim,device=args.device,dtype=dtype)
    zeros=torch.zeros(1,32,action_dim,device=args.device,dtype=dtype)
    def variation(x):
        return {'max_abs_delta_from_first_token':(x-x[:,:1]).abs().max().item(),
                'mean_temporal_std':x.float().std(dim=1).mean().item()}
    rows=[]
    with torch.no_grad():
        for scale in [0.0,0.1]:
            model.action_position_scale=scale
            t0=torch.tensor([1000.],device=args.device,dtype=dtype)
            t1=torch.tensor([100.],device=args.device,dtype=dtype)
            coarse=model(zeros,bridges,t0,context=context)
            refined=model(.9*coarse,bridges,t1,context=context)
            noisy=model(torch.randn_like(zeros),bridges,t0,context=context)
            rows.append(dict(position_scale=scale,coarse=variation(coarse),refined=variation(refined),
                             random_action_input=variation(noisy)))
    result=dict(checkpoint=str(args.checkpoint),checkpoint_bytes=args.checkpoint.stat().st_size,
        seed=42,dtype=args.dtype,device=args.device,num_heads=args.num_heads,video_layers=args.video_layers,
        bridge_tokens=64,action_tokens=32,synthetic_bridges=True,results=rows,
        caveat='Synthetic features, trained action weights. No inference-success claim for untrained position encoding.')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(rows,indent=2))

if __name__=='__main__':main()
