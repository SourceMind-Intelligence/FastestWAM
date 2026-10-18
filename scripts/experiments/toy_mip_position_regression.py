#!/usr/bin/env python3
"""Synthetic conditional trajectory regression: zero-input cross-attention control.

Not a robotics benchmark. Measures the temporal-constant expressivity limit.
"""
import argparse,json,math,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import torch
from openwam.model.action_backbone.separate_action_dit import ActionDiT


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--steps',type=int,default=500)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();torch.set_num_threads(2)
    results=[]
    for seed in [0,1,2]:
        torch.manual_seed(seed)
        features=torch.randn(192,1,32)
        phase=torch.arange(32)*2*math.pi/32
        amplitude=.5+.3*features[:,0,0].tanh()
        offset=.2*features[:,0,1].tanh()
        targets=torch.stack([amplitude[:,None]*phase.sin()+offset[:,None],
                             amplitude[:,None]*phase.cos()-offset[:,None]],dim=-1)
        inputs=torch.zeros(192,32,2)
        oracle=((targets[128:]-targets[128:].mean(1,keepdim=True))**2).mean().item()
        batches=torch.randint(128,(args.steps,16))
        init=None
        for scale in [0.0,0.1]:
            model=ActionDiT(action_dim=2,dim=32,ffn_dim=64,num_heads=4,num_layers=2,
                video_dim=32,bridge_layers=(0,1),text_dim=32,action_position_scale=scale)
            if init is None:init={k:v.detach().clone() for k,v in model.state_dict().items()}
            else:model.load_state_dict(init)
            optimizer=torch.optim.AdamW(model.parameters(),lr=1e-3)
            history=[]
            for step,ix in enumerate(batches):
                optimizer.zero_grad()
                pred=model(inputs[ix],{0:features[ix],1:features[ix]},torch.full((len(ix),),1000.))
                loss=(pred-targets[ix]).square().mean();loss.backward();optimizer.step()
                if step%50==0:history.append(dict(step=step+1,loss=loss.item()))
            model.eval()
            with torch.no_grad():
                pred=model(inputs[128:],{0:features[128:],1:features[128:]},torch.full((64,),1000.))
                error=(pred-targets[128:]).square().mean().item()
            results.append(dict(seed=seed,position_scale=scale,steps=args.steps,train_samples=128,
                test_samples=64,test_mse=error,oracle_constant_mse=oracle,
                mean_temporal_std=pred.std(1).mean().item(),history=history))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(dict(task='Synthetic sine/cosine trajectory conditioned on observation',
        note='Coarse regression only, no video model, no closed-loop robot evaluation.',results=results),indent=2)+'\n')
    print(json.dumps([{k:v for k,v in r.items() if k!='history'} for r in results],indent=2))

if __name__=='__main__':main()
