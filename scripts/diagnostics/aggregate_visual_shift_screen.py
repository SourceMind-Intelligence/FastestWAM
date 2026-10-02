"""Audit matched LIBERO-Plus 103-task screens from raw one-rollout records."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path


def load_run(root: Path) -> tuple[dict, dict[tuple[str, int], dict]]:
    manifest = json.loads((root / 'manifest.json').read_text())
    summary = json.loads((root / 'summary.json').read_text())
    overall = summary['overall']
    assert (overall['tasks_complete'], overall['tasks_expected'], overall['trials']) == (103, 103, 103), overall
    assert not summary['missing_runs'], summary['missing_runs']
    rows = {}
    for row in summary['task_results']:
        key = (row['suite'], int(row['task_id']))
        assert key not in rows, key
        assert row['complete'] and row['trials'] == 1 and len(row['result_paths']) == 1, row
        path = Path(row['result_paths'][0])
        if not path.is_file():
            matches = list((root / 'videos' / key[0] / f'task_{key[1]:02d}' / 'trials_000_000').glob('attempt_*/results.json'))
            assert len(matches) == 1, (path, matches)
            path = matches[0]
        raw = json.loads(path.read_text())
        assert raw['suite'] == key[0] and int(raw['task_id']) == key[1], (path, key)
        assert (raw['trial_start'], raw['trial_stop']) == (0, 1), (path, raw)
        assert len(raw['trials']) == 1 and int(raw['trials'][0]['trial']) == 0, path
        success = bool(raw['trials'][0]['success'])
        assert raw['successes'] == row['successes'] == int(success), (path, row)
        rows[key] = {
            'success': success,
            'suite': key[0],
            'task_id': key[1],
            'name': row['task_name'],
            'category': row['category'],
            'difficulty_level': int(row['difficulty_level']),
            'result_path': str(path),
        }
    assert len(rows) == 103 and sum(x['success'] for x in rows.values()) == overall['successes']
    jobs_payload = [{'suite': key[0], 'task_id': key[1]} for key in rows]
    jobs_hash = hashlib.sha256(json.dumps(jobs_payload, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    assert jobs_hash == manifest['resume_signature']['jobs_sha256'], (jobs_hash, manifest['resume_signature']['jobs_sha256'])
    return manifest, rows


def group(keys: list[tuple[str, int]], runs: dict[str, dict]) -> dict:
    result = {'tasks': len(keys)}
    for label, rows in runs.items():
        result[f'{label}_successes'] = sum(rows[key]['success'] for key in keys)
    for left, right in (('A', 'C'), ('A', 'released'), ('C', 'released')):
        if left in runs and right in runs:
            result[f'{left}_only_vs_{right}'] = sum(
                runs[left][key]['success'] and not runs[right][key]['success'] for key in keys
            )
            result[f'{right}_only_vs_{left}'] = sum(
                runs[right][key]['success'] and not runs[left][key]['success'] for key in keys
            )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--a', type=Path, required=True)
    parser.add_argument('--c', type=Path, required=True)
    parser.add_argument('--released', type=Path)
    parser.add_argument('--sample-plan', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()

    roots = {'A': args.a, 'C': args.c}
    if args.released:
        roots['released'] = args.released
    manifests = {}
    runs = {}
    for label, root in roots.items():
        manifests[label], runs[label] = load_run(root)
    signatures = {label: manifest['resume_signature'] for label, manifest in manifests.items()}
    first = signatures['A']
    for label, signature in signatures.items():
        for field in (
            'protocol_version', 'policy_config_sha256', 'seed_override', 'effective_seed',
            'mujoco_version', 'inference_mode', 'inference_horizon', 'denoise_mode',
            'trial_start', 'num_trials', 'jobs_sha256', 'jobs_count', 'render_gpus',
        ):
            assert signature[field] == first[field], (label, field, signature[field], first[field])
        assert set(runs[label]) == set(runs['A']), label
    assert first['jobs_count'] == 103 and first['num_trials'] == 1
    assert signatures['A']['denoise_steps'] == 2
    assert signatures['C']['denoise_steps'] == 10  # C MIP still executes two fixed action passes.
    if 'released' in signatures:
        assert signatures['released']['denoise_steps'] == 2

    plan = json.loads(args.sample_plan.read_text())
    assert plan['sample_ratio'] == 0.01 and isinstance(plan['sample_seed'], int)
    planned = {(row['suite'], int(row['task_id'])): row for row in plan['tasks']}
    assert len(planned) == 103 and set(planned) == set(runs['A'])
    for key, expected in planned.items():
        for label, rows in runs.items():
            actual = rows[key]
            assert (
                actual['name'] == expected['name']
                and actual['category'] == expected['category']
                and actual['difficulty_level'] == int(expected['difficulty_level'])
            ), (label, key, actual, expected)

    keys = sorted(planned)
    by_suite = defaultdict(list)
    by_category = defaultdict(list)
    by_difficulty = defaultdict(list)
    for key in keys:
        meta = planned[key]
        by_suite[key[0]].append(key)
        by_category[meta['category']].append(key)
        by_difficulty[str(meta['difficulty_level'])].append(key)
    payload = {
        'scope': 'Reproducible 1% LIBERO-Plus screen, 103 tasks, one rollout each; not the official full 10,030-task score',
        'note': 'A/C checkpoints share training foundation/data/seed but differ in objectives and training cost. Released checkpoint has a different training history. Small group counts are descriptive.',
        'sample_ratio': plan['sample_ratio'],
        'sample_seed': plan['sample_seed'],
        'jobs_sha256': first['jobs_sha256'],
        'checkpoints': {label: signature['checkpoint'] for label, signature in signatures.items()},
        'overall': group(keys, runs),
        'suites': {name: group(items, runs) for name, items in sorted(by_suite.items())},
        'categories': {name: group(items, runs) for name, items in sorted(by_category.items())},
        'difficulty_levels': {name: group(items, runs) for name, items in sorted(by_difficulty.items())},
        'tasks': [
            {
                **{field: planned[key][field] for field in ('suite', 'task_id', 'name', 'category', 'difficulty_level')},
                **{f'{label}_success': runs[label][key]['success'] for label in runs},
                **{f'{label}_source': runs[label][key]['result_path'] for label in runs},
            }
            for key in keys
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + '\n')
    print(json.dumps({'overall': payload['overall'], 'suites': payload['suites']}, indent=2))


if __name__ == '__main__':
    main()
