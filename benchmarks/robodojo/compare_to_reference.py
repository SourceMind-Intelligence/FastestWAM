#!/usr/bin/env python3
"""Compare a RoboDojo eval run against the published OpenWAM-α per-seed scores.

Reads an XPolicyLab/RoboDojo eval tree (the layout RoboDojo's
``scripts/internal/summarize_result.py`` reads):

    <root>/<task>/<policy>/<embodiment>/<seed>_ckpt_name=.../<timestamp>/_result.json

and aggregates it with the same rules (latest timestamp per seed; Generalization
tasks merge the first 25 standard + first 25 ``_random`` episodes; everything
else takes the first 50; Avg = mean of the five dimension means). Each task is
then compared to ``assets/robodojo_verification/openwam_robodojo_per_seed.json``
with a pooled two-proportion z-test over all seeds present, so "within noise"
has a concrete meaning: |z| <= --z (default 3.0, roughly a 5% family-wise false
alarm rate over 42 tasks).

    python benchmarks/robodojo/compare_to_reference.py /path/to/eval_result/RoboDojo
    python benchmarks/robodojo/compare_to_reference.py ROOT --policy OpenWAM --json out.json

Exit code is 0 when every complete task is within noise and the Avg SR is
within --avg-tol of the reference, 1 otherwise, 2 when nothing was found.
"""

import argparse
import json
import math
import os
import re
import statistics
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
REFERENCE = os.path.join(REPO, "assets", "robodojo_verification", "openwam_robodojo_per_seed.json")

DIMENSIONS = {
    "Generalization": [
        "stack_bowls", "push_T", "pack_objects_into_box", "fold_clothes",
        "hang_mugs", "sweep_blocks", "pour_liquid_into_cup", "make_toast",
        "arrange_largest_number", "sort_nesting_dolls_by_size",
        "store_laptop_and_headphones", "stack_blocks",
    ],
    "Precision": [
        "fasten_screws", "plug_in_charger", "insert_tubes",
        "pour_balls_into_vase", "play_Xylophone", "deposit_coin",
        "insert_key", "build_tower",
    ],
    "Long-Horizon": [
        "put_bottles_into_dustbin", "fill_pen_holder", "classify_objects",
        "play_tic_tac_toe", "fill_egg_holder", "organize_table",
        "make_kong", "play_stacking_toy",
    ],
    "Memory": [
        "cover_blocks", "match_and_pick_from_conveyor", "swap_blocks",
        "swap_T", "press_by_number", "imitate_sorting_sequence",
    ],
    "Open": [
        "align_blocks", "general_pickup", "stack_blocks_by_language",
        "solve_equation", "classify_objects_by_language",
        "pick_from_conveyor_by_image", "store_tools_in_toolbox",
        "pour_by_language",
    ],
}
GEN_TASKS = set(DIMENSIONS["Generalization"])
SEEDS = [0, 1, 2]
STANDALONE, HALF = 50, 25
SEED_RE = re.compile(r"^(\d+)_")


def subdirs(path):
    if not os.path.isdir(path):
        return []
    return sorted(n for n in os.listdir(path) if os.path.isdir(os.path.join(path, n)))


def load_entries(ts_dir):
    try:
        with open(os.path.join(ts_dir, "_result.json")) as fh:
            details = json.load(fh).get("details", {})
    except (json.JSONDecodeError, OSError):
        return None
    items = []
    for key, entry in details.items():
        try:
            layout = int(key)
        except (TypeError, ValueError):
            continue
        items.append((layout, bool(entry.get("success", False)), float(entry.get("score", 0.0) or 0.0)))
    items.sort()
    return [(s, sc) for _, s, sc in items]


def scan(root, task, policy):
    """{seed: entries} for the latest timestamp of each seed of one policy."""
    out, latest = {}, {}
    pol_dir = os.path.join(root, task, policy)
    for emb in subdirs(pol_dir):
        for run in subdirs(os.path.join(pol_dir, emb)):
            m = SEED_RE.match(run)
            if not m:
                continue
            seed = int(m.group(1))
            run_dir = os.path.join(pol_dir, emb, run)
            for ts in subdirs(run_dir):
                if not os.path.isfile(os.path.join(run_dir, ts, "_result.json")):
                    continue
                if seed in latest and ts <= latest[seed]:
                    continue
                entries = load_entries(os.path.join(run_dir, ts))
                if entries is not None:
                    out[seed], latest[seed] = entries, ts
    return out


def stats(entries):
    n = len(entries)
    return sum(s for s, _ in entries), sum(sc for _, sc in entries) / n * 100.0, n


def collect(root, policy):
    """{task: {seed: (successes, score_pct, n)}} for complete (task, seed) cells."""
    cells = {}
    for dim_tasks in DIMENSIONS.values():
        for task in dim_tasks:
            base = scan(root, task, policy)
            rand = scan(root, task + "_random", policy) if task in GEN_TASKS else {}
            for seed in SEEDS:
                if task in GEN_TASKS:
                    std, rnd = base.get(seed, []), rand.get(seed, [])
                    if len(std) < HALF or len(rnd) < HALF:
                        continue
                    entries = std[:HALF] + rnd[:HALF]
                else:
                    entries = base.get(seed, [])
                    if len(entries) < STANDALONE:
                        continue
                    entries = entries[:STANDALONE]
                cells.setdefault(task, {})[seed] = stats(entries)
    return cells


def reference():
    with open(REFERENCE) as fh:
        ref = json.load(fh)["merged_all_tasks"]
    return {t: {s: (v["sr"][i], v["score"][i]) for i, s in enumerate(SEEDS)} for t, v in ref.items()}


def z_score(obs_k, obs_n, ref_k, ref_n):
    p = (obs_k + ref_k) / (obs_n + ref_n)
    se = math.sqrt(p * (1 - p) * (1 / obs_n + 1 / ref_n))
    if se == 0:
        return 0.0
    return (obs_k / obs_n - ref_k / ref_n) / se


def seed_avg(per_task, seed):
    """Avg over five dimensions at one seed, or None if any task is missing."""
    dims = []
    for tasks in DIMENSIONS.values():
        vals = [per_task[t][seed] for t in tasks if t in per_task and seed in per_task[t]]
        if len(vals) != len(tasks):
            return None
        dims.append(statistics.fmean(vals))
    return statistics.fmean(dims)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", help="eval_result/RoboDojo directory")
    ap.add_argument("--policy", default="OpenWAM", help="policy directory name under each task (default: OpenWAM)")
    ap.add_argument("--z", type=float, default=3.0, help="per-task |z| threshold (default 3.0)")
    ap.add_argument("--avg-tol", type=float, default=1.5, help="allowed |Avg SR delta| in points (default 1.5)")
    ap.add_argument("--json", help="also write the comparison to this file")
    args = ap.parse_args()

    obs = collect(args.root, args.policy)
    if not obs:
        print(f"No complete (task, seed) cells for policy '{args.policy}' under {args.root}")
        return 2
    ref = reference()

    rows, flagged = [], []
    for dim, tasks in DIMENSIONS.items():
        for task in tasks:
            seeds = sorted(obs.get(task, {}))
            if not seeds:
                rows.append((dim, task, None))
                continue
            o_k = sum(obs[task][s][0] for s in seeds)
            o_n = sum(obs[task][s][2] for s in seeds)
            # Reference SRs are percentages of 50 episodes per seed.
            r_k = sum(round(ref[task][s][0] / 100 * STANDALONE) for s in seeds)
            r_n = STANDALONE * len(seeds)
            o_sr, r_sr = 100 * o_k / o_n, 100 * r_k / r_n
            o_sc = statistics.fmean(obs[task][s][1] for s in seeds)
            r_sc = statistics.fmean(ref[task][s][1] for s in seeds)
            z = z_score(o_k, o_n, r_k, r_n)
            row = dict(dim=dim, task=task, seeds=seeds, sr=o_sr, ref_sr=r_sr,
                       score=o_sc, ref_score=r_sc, z=z, ok=abs(z) <= args.z)
            rows.append((dim, task, row))
            if not row["ok"]:
                flagged.append(row)

    print(f"{'dimension':<15}{'task':<32}{'seeds':>7}{'SR':>8}{'ref':>8}{'dSR':>8}{'Sc':>8}{'ref':>8}{'z':>7}")
    for dim, task, r in rows:
        if r is None:
            print(f"{dim:<15}{task:<32}{'-':>7}  (no complete seeds)")
            continue
        mark = "" if r["ok"] else "  <-- outside noise"
        print(f"{dim:<15}{task:<32}{len(r['seeds']):>7}{r['sr']:>8.2f}{r['ref_sr']:>8.2f}"
              f"{r['sr'] - r['ref_sr']:>+8.2f}{r['score']:>8.2f}{r['ref_score']:>8.2f}{r['z']:>+7.2f}{mark}")

    obs_sr = {t: {s: 100 * c[0] / c[2] for s, c in v.items()} for t, v in obs.items()}
    ref_sr = {t: {s: v[s][0] for s in SEEDS} for t, v in ref.items()}
    obs_avgs = {s: seed_avg(obs_sr, s) for s in SEEDS}
    ref_avgs = {s: seed_avg(ref_sr, s) for s in SEEDS}
    done = [s for s in SEEDS if obs_avgs[s] is not None]
    complete = sum(len(v) for v in obs.values())
    print(f"\nComplete cells: {complete}/{42 * len(SEEDS)}")
    avg_ok = None
    if done:
        o = statistics.fmean(obs_avgs[s] for s in done)
        r = statistics.fmean(ref_avgs[s] for s in done)
        avg_ok = abs(o - r) <= args.avg_tol
        per_seed = ", ".join(f"s{s} {obs_avgs[s]:.2f} vs {ref_avgs[s]:.2f}" for s in done)
        print(f"Avg SR over seeds {done}: {o:.2f} vs reference {r:.2f} ({o - r:+.2f})  [{per_seed}]")
    else:
        print("Avg SR: needs all 42 tasks at one seed or more")
    print(f"Tasks outside noise (|z| > {args.z}): {len(flagged)}"
          + ("" if not flagged else " -> " + ", ".join(f["task"] for f in flagged)))

    if args.json:
        with open(args.json, "w") as fh:
            json.dump({"rows": [r for _, _, r in rows if r], "avg": {s: obs_avgs[s] for s in done},
                       "ref_avg": {s: ref_avgs[s] for s in done}, "flagged": [f["task"] for f in flagged]},
                      fh, indent=2)
    return 0 if not flagged and avg_ok is not False else 1


if __name__ == "__main__":
    sys.exit(main())
