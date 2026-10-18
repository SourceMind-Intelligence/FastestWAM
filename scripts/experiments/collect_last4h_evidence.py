#!/usr/bin/env python3
"""Snapshot completed tasks without counting unfinished/failed tasks as failures.

Safe to run during evaluation. Writes only results/last4h_20261018, never model
weights or source evaluation records. No network publication.
"""
import csv,gzip,hashlib,json,sys
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from benchmarks.libero import scheduler
from benchmarks.libero.run_all_suites import _native_read_result
scheduler._read_result=_native_read_result


def relative(x):
    if isinstance(x,dict):return {k:relative(v) for k,v in x.items()}
    if isinstance(x,list):return [relative(v) for v in x]
    if isinstance(x,str):return x.replace(str(ROOT)+'/', '')
    return x


def write(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(relative(data),indent=2)+'\n')


def main():
    dest=ROOT/'results/last4h_20261018';dest.mkdir(parents=True,exist_ok=True)
    status=[]
    for pointer in ['last4h_latest.txt','mip_position_rescue_latest.txt']:
        marker=ROOT/'outputs'/pointer
        if not marker.exists():continue
        source=Path(marker.read_text().strip())
        for mp in sorted(source.glob('*/manifest.json')):
            manifest=json.loads(mp.read_text());run=mp.parent
            jobs=[scheduler.TaskJob(**j) for j in manifest['initial_queue']]
            tr=manifest['trial_run'];trial_run=scheduler.TrialRun(tr['trial_start'],tr['num_trials'])
            summary,complete=scheduler._summarize(run,jobs,trial_run,write_files=False)
            target=dest/run.name
            write(target/'summary.json',summary);write(target/'manifest.json',manifest)
            episodes=[]
            for task in summary['task_results']:
                for path in task['result_paths']:
                    rp=Path(path);payload=json.loads(rp.read_text())
                    write(target/'tasks'/task['suite']/f"{task['task_id']:02d}.json",payload)
                    for e in payload['trials']:
                        episodes.append(dict(suite=task['suite'],task_id=task['task_id'],
                            trial=e['trial'],success=int(bool(e.get('success'))),
                            policy_steps=e.get('policy_steps')))
            with (target/'episodes.csv').open('w') as f:
                w=csv.DictWriter(f,fieldnames=['suite','task_id','trial','success','policy_steps'],lineterminator='\n')
                w.writeheader();w.writerows(episodes)
            row=dict(run=run.name,complete=complete,**summary['overall'])
            if run.name=='mot_fm_nfe1_seed42':
                row['excluded_reason']='Aborted after repeated EGL SIGABRT on GPU6; superseded by render2 run.'
            status.append(row)
    for marker_name,folder in [('flops_audit_latest.txt','flops'),('mip_position_rescue_latest.txt','rescue_training')]:
        marker=ROOT/'outputs'/marker_name
        if not marker.exists():continue
        source=Path(marker.read_text().strip())
        for mp in source.glob('*/20*/scaling_metrics.jsonl'):
            raw=mp.read_bytes();lines=raw.splitlines()
            # A running process can have a partially flushed final record.
            try: records=[json.loads(x) for x in lines]
            except json.JSONDecodeError:records=[json.loads(x) for x in lines[:-1]]
            if not records:continue
            out=dest/folder/mp.parent.parent.name;out.mkdir(parents=True,exist_ok=True)
            normalized=b'\n'.join(json.dumps(r).encode() for r in records)+b'\n'
            (out/'scaling_metrics.jsonl.gz').write_bytes(gzip.compress(normalized,mtime=0))
            config=mp.parent/'config.yaml'
            if config.exists():(out/'config.yaml').write_text(relative(config.read_text()))
            write(out/'summary.json',dict(records=len(records),first=records[0],last=records[-1],
                source=str(mp),snapshot_sha256=hashlib.sha256(normalized).hexdigest()))
    marker=ROOT/'outputs/latency_audit_latest.txt'
    if marker.exists():
        for p in Path(marker.read_text().strip()).glob('*.json'):
            write(dest/'latency'/p.name,json.loads(p.read_text()))
    write(dest/'status.json',dict(updated_at=datetime.now(timezone.utc).isoformat(),runs=status,
        note='All new evaluations are diagnostic. Rates cover completed tasks only. '
             'Incomplete and aborted runs must not be reported as full-benchmark scores.'))
    print(json.dumps(status,indent=2))

if __name__=='__main__':main()
