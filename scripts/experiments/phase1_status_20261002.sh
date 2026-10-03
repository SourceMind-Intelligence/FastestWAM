#!/usr/bin/env bash
# Read-only health check of the phase-1 runs on h100-box (shallow-head plan, 2026-10-02), on one screen:
# the queue, GPUs and disk; per run its step, losses, speed, ETA, checkpoints and log errors; then the
# LIBERO evals (finished or in progress).
set -uo pipefail
cd "${OPENWAM_ROOT:-/root/evan/Fastest-WAM-evan}" || exit 1
export PATH="/root/openwam-eval/env/bin:$PATH"
log_dir=logs/phase1-20261002
echo "time   $(date -u '+%Y-%m-%d %H:%M UTC'), tree $(git rev-parse --short HEAD 2> /dev/null) on $(git rev-parse --abbrev-ref HEAD 2> /dev/null)"
pid="$(cat "$log_dir/queue.pid" 2> /dev/null)"
if [[ -n "$pid" ]] && kill -0 "$pid" 2> /dev/null; then queue="running (pid $pid)"; else queue="not running"; fi
echo "queue  $queue; done: $(tr '\n' ' ' < "$log_dir/done.txt" 2> /dev/null)"
[[ -e "$log_dir/PAUSE" ]] && echo "queue  PAUSE file present"
tail -n 6 "$log_dir/status.txt" 2> /dev/null | sed 's/^/       /'
echo "gpus   $(nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader,nounits 2> /dev/null | awk -F', ' '{ printf "%s:%s%%/%.0fG ", $1, $2, $3 / 1024 }')"
echo "procs  $(pgrep -fc 'scripts/train.py') trainer, $(pgrep -fc 'run_all_suites.py') LIBERO eval"
echo "disk   $(df -h / | awk 'NR == 2 { print $4 " free of " $2 }')"
python3 - "$log_dir" << 'PY'
import glob
import json
import math
import os
import re
import sys
import time

log_dir = sys.argv[1]
now = time.time()
fixed_total = {"r1": 2000, "r3": 12000, "r4": 10000, "r5": 10000}
names = ["libero_dot_l1_20261002", "libero_dot_l2_20261002"]
names += [f"robodojo_alpha_{arm}_20261002" for arm in fixed_total]
loss_keys = ("loss", "loss_video", "loss_action", "loss_mip_t0", "loss_mip_t09")


def read_tail(path, size):
    try:
        with open(path, "rb") as f:
            f.seek(0, 2)
            end = f.tell()
            f.seek(max(0, end - size))
            text = f.read().decode("utf-8", "replace")
    except OSError:
        return []
    lines = text.splitlines()
    return lines[1:] if end > size else lines


def read_head(path, size):
    try:
        with open(path, "rb") as f:
            lines = f.read(size).decode("utf-8", "replace").splitlines()
    except OSError:
        return []
    return lines[:-1]


def rows_of(lines):
    rows = []
    for line in lines:
        try:
            rows.append(json.loads(line))
        except ValueError:
            pass
    return rows


def mean(rows, key):
    vals = [r[key] for r in rows if isinstance(r.get(key), (int, float)) and math.isfinite(r[key])]
    return sum(vals) / len(vals) if vals else None


def losses(rows):
    parts = []
    for key in loss_keys:
        val = mean(rows, key)
        if val is not None:
            parts.append(f"{key.replace('loss_', '')} {val:.4f}")
    return ", ".join(parts)


for name in names:
    dirs = sorted(d for d in glob.glob(f"outputs/openwam_checkpoints/{name}/*/") if os.path.isfile(d + "config.yaml"))
    if not dirs:
        continue
    run = dirs[-1].rstrip("/")
    metrics = os.path.join(run, "scaling_metrics.jsonl")
    head = rows_of(read_head(metrics, 200_000))[:200]
    tail = rows_of(read_tail(metrics, 400_000))[-200:]
    print(f"run    {name} ({os.path.basename(run)})")
    if not tail:
        print("       no steps logged yet")
    else:
        last = tail[-1]
        step = last["step"]
        sec_per_step = None
        if len(tail) > 1 and tail[-1]["step"] > tail[0]["step"]:
            sec_per_step = (tail[-1]["ts"] - tail[0]["ts"]) / (tail[-1]["step"] - tail[0]["step"])
        total = None
        arm = name.split("_")[2]
        if arm in fixed_total:
            total = fixed_total[arm]
        else:
            found = re.findall(r"(\d+)/(\d+) \[", "\n".join(read_tail(os.path.join(log_dir, name + ".log"), 400_000)))
            if found:
                total = int(found[-1][1])
        eta = ""
        if total and sec_per_step:
            eta = f", about {(total - step) * sec_per_step / 3600:.1f} h left"
        bad = sum(1 for r in tail for k in loss_keys if isinstance(r.get(k), float) and not math.isfinite(r[k]))
        speed = f"{sec_per_step:.2f} s/step" if sec_per_step else "s/step n/a"
        print(
            f"       step {step}/{total or '?'} (opt {last.get('opt_step')}, epoch {last.get('epoch')}), {speed}{eta}, "
            f"last row {(now - last['ts']) / 60:.0f} min ago"
        )
        print(f"       first 200 steps: {losses(head)}")
        print(f"       last 200 steps:  {losses(tail)}; grad_norm {mean(tail, 'grad_norm') or 0:.3f}; lr {last.get('lr'):.2e}")
        if bad:
            print(f"       NON-FINITE losses in the last 200 steps: {bad}")
    ckpts = sorted(glob.glob(run + "/checkpoint_step_*.safetensors"), key=lambda p: int(re.findall(r"\d+", os.path.basename(p))[-1]))
    states = sorted(glob.glob(run + "/accel_state_step_*/trainer_state.json"))
    if ckpts:
        newest = ckpts[-1]
        age = (now - os.path.getmtime(newest)) / 3600
        print(f"       weights {', '.join(os.path.basename(p)[16:-12] for p in ckpts)} (newest {age:.1f} h ago); full states {len(states)}")
    log_lines = read_tail(os.path.join(log_dir, name + ".log"), 200_000)
    errors = [ln for ln in log_lines if re.search(r"Traceback|Error|out of memory|NCCL.*(timeout|error)", ln)]
    if errors:
        print(f"       log errors: {len(errors)}; last: {errors[-1].strip()[:200]}")

for out in sorted(glob.glob("outputs/libero/phase1_20261002/*/")):
    tag = os.path.basename(out.rstrip("/"))
    summary = os.path.join(out, "summary.json")
    if os.path.isfile(summary):
        data = json.load(open(summary))
        total = data["overall"]
        suites = ", ".join(f"{r['suite'].replace('libero_', '')} {r['successes']}/{r['trials']}" for r in data["suite_results"])
        t06 = [r for r in data["task_results"] if r["suite"] == "libero_10" and r["task_id"] == 6]
        t06_text = f"; long task 06 {t06[0]['successes']}/{t06[0]['trials']}" if t06 else ""
        print(f"eval   {tag}: {total['successes']}/{total['trials']} ({total['tasks_complete']}/{total['tasks_expected']} tasks; {suites}{t06_text})")
        continue
    done = {}
    for path in glob.glob(out + "videos/*/task_*/*/*/results.json"):
        try:
            result = json.load(open(path))
        except (OSError, ValueError):
            continue
        done[(result["suite"], result["task_id"])] = (result["successes"], len(result.get("trials", [])))
    wins = sum(s for s, _ in done.values())
    trials = sum(t for _, t in done.values())
    print(f"eval   {tag}: in progress, {len(done)} tasks done, {wins}/{trials} so far")
PY
