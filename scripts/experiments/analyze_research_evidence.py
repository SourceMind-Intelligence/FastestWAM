#!/usr/bin/env python3
"""Task-cluster bootstrap and training-loss diagnostics for the evidence archive."""
import argparse
import gzip
import json
from pathlib import Path
import statistics
import numpy as np


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path('results/research_audit_20261018'))
    args=p.parse_args();root=args.root
    pairs=[('sf_decision_A_fm_2step_all_suites_step42730_20260927',
            'sf_decision_C_mip_mixed_long_suite_step42730_20260927'),
           ('dot_FM_flow_final_2step','dot8_FM_flow_2step')]
    comparisons=[]
    for a,b in pairs:
        def load(name):
            rows=json.loads((root/name/'summary.json').read_text())['task_results']
            return {(r['suite'],r['task_id']):r['successes']/r['trials'] for r in rows}
        x,y=load(a),load(b)
        if x.keys()!=y.keys():
            raise ValueError('Task sets differ')
        differences=np.array([y[k]-x[k] for k in sorted(x)])*100
        bootstrap=np.random.default_rng(42).choice(differences,(20000,len(differences))).mean(axis=1)
        comparisons.append(dict(baseline=a,candidate=b,delta_pp=float(differences.mean()),
            task_bootstrap_95ci_pp=np.quantile(bootstrap,[.025,.975]).tolist(),
            bootstrap_seed=42,resamples=20000,
            caveat='Tasks resampled jointly across arms. Does not measure training-seed uncertainty.'))
    losses={}
    for path in sorted(root.glob('*/scaling_metrics.jsonl.gz')):
        records=[json.loads(line) for line in gzip.decompress(path.read_bytes()).splitlines()]
        losses[path.parent.name]={k:statistics.mean(r[k] for r in records[-100:] if r.get(k) is not None)
            for k in ['loss_action','pred0_gt_mse','pred1_gt_mse']
            if any(r.get(k) is not None for r in records[-100:])}
    (root/'statistics.json').write_text(json.dumps(dict(comparisons=comparisons,
        last100_training_loss=losses),indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    runs=json.loads((root/'runs.json').read_text());rates={r['run']:r['success_pct'] for r in runs}
    keys=['sf_decision_A_fm_2step_all_suites_step42730_20260927',
          'sf_decision_C_mip_mixed_long_suite_step42730_20260927',
          'dot_FM_flow_final_2step','dot8_FM_flow_2step',
          'dot4_MIP_gt_from_foundation_2step','dot_SF_from_FM_freeze_2step']
    names=['MoT FM','MoT Mixed','Cross-4 FM','Cross-8 FM','Cross-4 MIP GT','Cross-4 Mixed FT']
    fig,axes=plt.subplots(1,2,figsize=(12,4.8),layout='constrained')
    bars=axes[0].barh(names,[rates[k] for k in keys],color=['#457b9d','#457b9d','#457b9d','#457b9d','#c15c45','#c15c45'])
    axes[0].bar_label(bars,fmt='%.2f%%',padding=3,fontsize=9)
    axes[0].set_xlim(0,112);axes[0].invert_yaxis();axes[0].set_xlabel('LIBERO success (%) — 2000 episodes each')
    axes[0].set_title('Observed outcomes (training budgets differ)')
    lk=['dot4_MIP_gt_from_foundation_2step','dot_MIP_gt_from_FM_2step','dot_SF_from_FM_freeze_2step']
    indices=np.arange(len(lk));width=.35
    axes[1].bar(indices-width/2,[losses[k]['pred0_gt_mse'] for k in lk],width,label='Coarse / zero input',color='#c15c45')
    axes[1].bar(indices+width/2,[losses[k]['pred1_gt_mse'] for k in lk],width,label='Refine / training input',color='#457b9d')
    axes[1].set_xticks(indices,['MIP GT','FM → GT','FM → Mixed']);axes[1].set_yscale('log')
    axes[1].set_ylabel('Masked training MSE (last 100 microbatches)')
    axes[1].set_title('Low refine loss hides poor coarse prediction');axes[1].legend(fontsize=9)
    fig.savefig(root/'evidence_overview.png',dpi=180)
    plt.close(fig)
    print(json.dumps(comparisons,indent=2))

if __name__=='__main__':main()
