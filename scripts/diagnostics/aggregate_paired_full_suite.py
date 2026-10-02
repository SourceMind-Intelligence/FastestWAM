"""Audit and aggregate matched A FM-2 / C mixed-MIP LIBERO result files."""

import argparse
import json
import math
import statistics
from pathlib import Path


SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
TRIALS = tuple(range(50))


def load_run(root: Path):
    manifest = json.loads((root / "manifest.json").read_text())
    signature = manifest["resume_signature"]
    results = {}
    files = sorted(root.glob("videos/*/task_*/trials_000_049/attempt_*/results.json"))
    for path in files:
        raw = json.loads(path.read_text())
        suite, task_id = raw["suite"], int(raw["task_id"])
        key = (suite, task_id)
        trials = {int(row["trial"]): bool(row["success"]) for row in raw["trials"]}
        if tuple(sorted(trials)) != TRIALS or sum(trials.values()) != raw["successes"]:
            raise ValueError(f"Incomplete or inconsistent trials: {path}")
        if key in results:
            raise ValueError(f"Multiple complete attempts for {key}: {results[key]['path']} and {path}")
        results[key] = {"trials": trials, "path": str(path), "successes": raw["successes"]}
    expected = {(suite, task_id) for suite in SUITES for task_id in range(10)}
    if set(results) != expected:
        missing = sorted(expected - set(results))
        extra = sorted(set(results) - expected)
        raise ValueError(f"Expected 40 complete tasks at {root}; missing={missing}, extra={extra}")
    return manifest, signature, results


def exact_mcnemar(a_only: int, c_only: int) -> float:
    n = a_only + c_only
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(min(a_only, c_only) + 1)) / 2**n)


def aggregate(keys, a, c):
    a_success = c_success = a_only = c_only = 0
    task_differences = []
    for key in keys:
        aa, cc = a[key]["trials"], c[key]["trials"]
        task_differences.append(a[key]["successes"] - c[key]["successes"])
        for trial in TRIALS:
            av, cv = aa[trial], cc[trial]
            a_success += av
            c_success += cv
            a_only += av and not cv
            c_only += cv and not av
    return {
        "tasks": len(keys),
        "trials": len(keys) * len(TRIALS),
        "a_successes": a_success,
        "c_successes": c_success,
        "a_minus_c": a_success - c_success,
        "a_only": a_only,
        "c_only": c_only,
        "tasks_a_better": sum(diff > 0 for diff in task_differences),
        "tasks_c_better": sum(diff < 0 for diff in task_differences),
        "tasks_tied": sum(diff == 0 for diff in task_differences),
        "median_task_difference": statistics.median(task_differences),
        "mcnemar_exact_two_sided_descriptive": exact_mcnemar(a_only, c_only),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("a_root", type=Path)
    parser.add_argument("c_root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    a_manifest, a_sig, a = load_run(args.a_root)
    c_manifest, c_sig, c = load_run(args.c_root)
    for field in (
        "protocol_version",
        "policy_config_sha256",
        "effective_seed",
        "mujoco_version",
        "inference_mode",
        "inference_horizon",
        "trial_start",
        "num_trials",
        "jobs_sha256",
        "jobs_count",
    ):
        if a_sig[field] != c_sig[field]:
            raise ValueError(f"Protocol differs on {field}: {a_sig[field]!r} versus {c_sig[field]!r}")
    if a_sig["jobs_count"] != 40:
        raise ValueError("Expected all 40 tasks in each final manifest")
    keys = sorted(a)
    payload = {
        "a_checkpoint": a_manifest["checkpoint"],
        "c_checkpoint": c_manifest["checkpoint"],
        "a_denoise_steps_flag": a_sig["denoise_steps"],
        "c_denoise_steps_flag": c_sig["denoise_steps"],
        "note": "C uses a fixed two-pass MIP action path; its denoise_steps flag does not set MIP passes. Task win/tie counts show consistency across benchmark tasks. Exact paired trial p-values are descriptive because trials are clustered within tasks and the benchmark tasks are fixed.",
        "overall": aggregate(keys, a, c),
        "suites": {
            suite: aggregate([key for key in keys if key[0] == suite], a, c)
            for suite in SUITES
        },
        "tasks": [
            {
                "suite": suite,
                "task_id": task_id,
                **aggregate([(suite, task_id)], a, c),
                "a_source": a[(suite, task_id)]["path"],
                "c_source": c[(suite, task_id)]["path"],
            }
            for suite, task_id in keys
        ],
    }
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"overall": payload["overall"], "suites": payload["suites"]}, indent=2))


if __name__ == "__main__":
    main()
