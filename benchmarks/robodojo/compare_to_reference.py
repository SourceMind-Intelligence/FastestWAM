#!/usr/bin/env python3
"""Compare collected RoboDojo cells with the published OpenWAM-alpha per-seed reference.

  compare_to_reference.py <results dir> [--reference JSON] [--md FILE]

The reference defaults to assets/robodojo_verification/openwam_robodojo_per_seed.json.
Reads every cells/<task>__s<seed>/meta.json (written by runpod/pod/cell.sh: success_rate, score, eval_time, driver). A paired
Generalization task is reported as the leaderboard does: (standard + _random) / 2 per seed, and only for seeds where
both halves finished. Task means are over the seeds that finished; the averages are over the tasks that have any
finished seed, with the reference averaged over the same tasks and seeds so a partial run compares like with like.
The overall average is the mean of the five dimension means, as on the leaderboard (11.92 / 17.18).
"""

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def mean(xs):
    return sum(xs) / len(xs) if xs else None


def fmt(x):
    return "   -  " if x is None else f"{x:6.2f}"


def main():
    args = sys.argv[1:]
    ref_path, md = HERE.parents[1] / "assets/robodojo_verification/openwam_robodojo_per_seed.json", None
    for flag in ("--reference", "--md"):
        if flag in args:
            i = args.index(flag)
            value = args[i + 1]
            del args[i : i + 2]
            if flag == "--md":
                md = Path(value)
            else:
                ref_path = Path(value)
    if len(args) != 1:
        sys.exit(__doc__)
    ref = json.loads(Path(ref_path).read_text())
    inv = json.loads((HERE / "tasks.json").read_text())
    dim_of = {t: d for d, ts in inv["dimensions"].items() for t in ts}
    paired = set(inv["paired_random"])
    seeds = ref["seeds"]
    cells, drivers, short = {}, set(), []
    for p in sorted(Path(args[0]).glob("cells/*/meta.json")):
        m = json.loads(p.read_text())
        if m.get("runner_exit") != 0 or m.get("success_rate") is None:
            continue
        cells[(m["task"], m["seed"])] = (m["success_rate"] * 100, m["score"])
        drivers.add(m["gpu"].split(", ")[1])
        half = m["task"].removesuffix("_random") in paired
        want = ref["episodes"]["paired_half" if half else "standalone_task"]
        if m.get("eval_num", "native") == "native" and m["eval_time"] != want:
            short.append(f"{m['cell']} ran {m['eval_time']} episodes, protocol {want}")
    rows, lines = [], []
    for task, r in ref["merged_all_tasks"].items():
        ours_sr, ours_sc, ref_sr, ref_sc, done = [], [], [], [], []
        for i, s in enumerate(seeds):
            parts = [cells.get((task, s))] + ([cells.get((task + "_random", s))] if task in paired else [])
            if any(p is None for p in parts):
                continue
            ours_sr.append(mean([p[0] for p in parts]))
            ours_sc.append(mean([p[1] for p in parts]))
            ref_sr.append(r["sr"][i])
            ref_sc.append(r["score"][i])
            done.append(s)
        rows.append(
            dict(
                task=task,
                dim=dim_of[task],
                seeds=done,
                sr=mean(ours_sr),
                sc=mean(ours_sc),
                rsr=mean(ref_sr),
                rsc=mean(ref_sc),
                full_rsr=mean(r["sr"]),
                full_rsc=mean(r["score"]),
                per_seed=ours_sr,
                ref_per_seed=r["sr"],
            )
        )
    head = f"{'task':34s} {'dim':14s} {'seeds':6s} {'SR':>6s} {'ref':>6s} {'d':>6s}  {'Score':>6s} {'ref':>6s} {'d':>6s}  per-seed SR ours | ref"
    lines += [head, "-" * len(head)]
    for x in sorted(rows, key=lambda x: (x["dim"], x["task"])):
        d_sr = None if x["sr"] is None else x["sr"] - x["rsr"]
        d_sc = None if x["sc"] is None else x["sc"] - x["rsc"]
        ours = "/".join(f"{v:g}" for v in x["per_seed"]) or "-"
        lines.append(
            f"{x['task']:34s} {x['dim']:14s} {','.join(map(str, x['seeds'])) or '-':6s} {fmt(x['sr'])} {fmt(x['rsr'])} {fmt(d_sr)}  "
            f"{fmt(x['sc'])} {fmt(x['rsc'])} {fmt(d_sc)}  {ours} | {'/'.join(f'{v:g}' for v in x['ref_per_seed'])}"
        )
    have = [x for x in rows if x["sr"] is not None]
    complete = [x for x in rows if len(x["seeds"]) == len(seeds)]
    lines.append("")
    lines.append(
        f"cells finished: {len(cells)}; tasks with a finished seed: {len(have)}/42; tasks with all {len(seeds)} seeds: {len(complete)}/42; drivers: {', '.join(sorted(drivers)) or '-'}"
    )
    dim_means = []
    for dim in inv["dimensions"]:
        sub = [x for x in have if x["dim"] == dim]
        allsub = [x for x in rows if x["dim"] == dim]
        if not sub:
            continue
        key = {"generalization": "generalization_merged", "long-horizon": "long_horizon"}.get(dim, dim)
        pub = ref["published_aggregates"][key]
        partial = not (len(sub) == len(allsub) and all(len(x["seeds"]) == len(seeds) for x in sub))
        m = [mean([x[k] for x in sub]) for k in ("sr", "rsr", "sc", "rsc")]
        dim_means.append((m, partial))
        lines.append(
            f"{dim:14s} tasks {len(sub):2d}/{len(allsub):2d}  SR {fmt(m[0])} vs ref {fmt(m[1])}   Score {fmt(m[2])} vs ref {fmt(m[3])}   "
            f"published {pub['sr']}/{pub['score']}{'  (partial: reference over the same cells)' if partial else ''}"
        )
    if dim_means:  # the leaderboard average is the mean of the five dimension means, not of the 42 tasks
        m = [mean([d[0][i] for d in dim_means]) for i in range(4)]
        pub = ref["published_aggregates"]["average"]
        partial = len(dim_means) < len(inv["dimensions"]) or any(d[1] for d in dim_means)
        lines.append(
            f"{'average':14s} dims  {len(dim_means)}/{len(inv['dimensions'])}   SR {fmt(m[0])} vs ref {fmt(m[1])}   Score {fmt(m[2])} vs ref {fmt(m[3])}   "
            f"published {pub['sr']}/{pub['score']}{'  (partial: reference over the same cells)' if partial else ''}"
        )
    lines += [f"WARNING: {s}" for s in short]
    text = "\n".join(lines)
    print(text)
    if md:
        md.write_text("```\n" + text + "\n```\n")


if __name__ == "__main__":
    main()
