#!/usr/bin/env python3
"""Archive small, auditable evaluation evidence without checkpoints or secrets.

Usage: python scripts/experiments/export_research_evidence.py
Also accepts --source (repeatable) and --output for additional evaluation roots.
"""
from __future__ import annotations
import argparse
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def relative(value):
    if isinstance(value, dict):
        return {k: relative(v) for k, v in value.items()}
    if isinstance(value, list):
        return [relative(v) for v in value]
    if isinstance(value, str):
        return value.replace(str(ROOT) + '/', '')
    return value

def write_json(path, obj):
    path.write_text(json.dumps(relative(obj), indent=2, allow_nan=False) + '\n')

def wilson(a, n):
    if not n:
        return [None, None]
    p, z = a / n, 1.959963984540054
    center = (p + z*z/(2*n)) / (1+z*z/n)
    radius = z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/(1+z*z/n)
    return [100*(center-radius), 100*(center+radius)]

def export(source, output):
    rows = []
    for sp in sorted(source.glob('*/summary.json')):
        original = json.loads(sp.read_text())
        tasks = original.get('task_results', [])
        if not tasks:
            continue
        dest = output / sp.parent.name
        dest.mkdir(parents=True, exist_ok=True)
        mp = sp.parent / 'manifest.json'
        manifest = json.loads(mp.read_text()) if mp.exists() else {}
        write_json(dest/'summary.json', original)
        write_json(dest/'manifest.json', manifest)
        checkpoint = Path(manifest.get('checkpoint', ''))
        if not checkpoint.is_absolute():
            checkpoint = ROOT / checkpoint
        config_path = checkpoint.parent / 'config.yaml'
        if config_path.is_file():
            # Configs contain no environment credentials; .env is never read.
            (dest/'training_config.yaml').write_text(relative(config_path.read_text()))
        metric_path = checkpoint.parent / 'scaling_metrics.jsonl'
        if metric_path.is_file():
            raw = metric_path.read_bytes()
            (dest/'scaling_metrics.jsonl.gz').write_bytes(gzip.compress(raw, mtime=0))
            records = [json.loads(line) for line in raw.splitlines() if line.strip()]
            if records:
                write_json(dest/'training_audit.json', dict(
                    source=str(metric_path), sha256=sha(metric_path),
                    config_sha256=sha(config_path) if config_path.exists() else None,
                    first=records[0], last=records[-1], records=len(records),
                    elapsed_hours=(records[-1]['ts']-records[0]['ts'])/3600,
                    nonzero_flops_records=sum(bool(r.get('step_tflops')) for r in records),
                    note='step/global_batch refer to microbatches; opt_step counts updates. '
                         'Elapsed time includes any pauses between logged records.'))
        errors, episodes = [], []
        for t in tasks:
            seen = {}
            for name in t.get('result_paths', []):
                path = Path(name)
                if not path.is_absolute():
                    path = ROOT/path
                if not path.exists():
                    errors.append(f'missing result: {name}')
                    continue
                result = json.loads(path.read_text())
                for e in result.get('trials', []):
                    key = e['trial']
                    if key in seen:
                        errors.append(f'duplicate trial {t["suite"]}/{t["task_id"]}/{key}')
                    seen[key] = e
            if len(seen) != t['trials'] or sum(bool(e.get('success')) for e in seen.values()) != t['successes']:
                errors.append(f'counts mismatch: {t["suite"]}/{t["task_id"]}')
            for trial, e in sorted(seen.items()):
                episodes.append(dict(suite=t['suite'], task_id=t['task_id'], trial=trial,
                    success=int(bool(e.get('success'))), policy_steps=e.get('policy_steps'),
                    error=e.get('error', '')))
        with (dest/'episodes.csv').open('w') as f:
            writer = csv.DictWriter(f, fieldnames=['suite','task_id','trial','success','policy_steps','error'], lineterminator='\n')
            writer.writeheader(); writer.writerows(episodes)
        a, n = sum(t['successes'] for t in tasks), sum(t['trials'] for t in tasks)
        full = (len({(t['suite'],t['task_id']) for t in tasks}) == 40
                and all(t.get('complete') and t['trials'] == 50 for t in tasks))
        sig = manifest.get('resume_signature', {})
        diagnostic = 'diagnostic' in sig.get('protocol_version', '')
        ci = wilson(a,n)
        row = dict(run=sp.parent.name,tasks=len(tasks),successes=a,trials=n,
            success_pct=round(100*a/n,4) if n else None,
            wilson_low_pct=ci[0],wilson_high_pct=ci[1],
            full_40x50=full and not diagnostic,raw_counts_verified=not errors,
            denoise_steps=manifest.get('denoise_steps'),seed=manifest.get('effective_seed'),
            checkpoint=relative(manifest.get('checkpoint')))
        for suite in ['libero_spatial','libero_object','libero_goal','libero_10']:
            subset=[t for t in tasks if t['suite']==suite]; count=sum(t['trials'] for t in subset)
            row[suite]=100*sum(t['successes'] for t in subset)/count if count else None
        write_json(dest/'audit.json',dict(summary_sha256=sha(sp),
            manifest_sha256=sha(mp) if mp.exists() else None,errors=errors,
            ci_note='Wilson interval treats episodes as independent; not training-seed or task-cluster uncertainty.',
            **row))
        rows.append(row)
    return rows

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,action='append')
    parser.add_argument('--output',type=Path,default=ROOT/'results/research_audit_20261018')
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    rows=[]
    for source in args.source or [ROOT/'outputs/libero']:
        rows.extend(export(source,args.output))
    if not rows:
        raise SystemExit('No completed task summaries found')
    with (args.output/'runs.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]), lineterminator='\n');writer.writeheader();writer.writerows(rows)
    write_json(args.output/'runs.json',rows)
    for r in rows:
        print(f"{r['run']}: {r['successes']}/{r['trials']} ({r['success_pct']}%) verified={r['raw_counts_verified']}")
    if any(not r['raw_counts_verified'] for r in rows):
        raise SystemExit('Some raw records are missing or inconsistent; see audit.json')
if __name__=='__main__':
    main()
