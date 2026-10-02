#!/usr/bin/env bash
# Pod side, inside tmux: one (task, seed) cell of the OpenWAM-alpha RoboDojo reproduction through RoboDojo's own
# runner (scripts/robodojo.sh benchmark, native episode counts, adapter eval_batch: true), then this pod stops itself.
#   cell.sh <task> <seed> [--out DIR] [--grace SECONDS] [--max-hours H] [--no-stop]
# Nothing is written into the shared checkouts on the volume: the runner is started from an overlay root (a directory
# of symlinks to $R/RoboDojo's entries on the container disk) whose eval_result/ and smoke_results/ live under --out.
#   --out DIR      results root (default $R/fastestwam-baseline; a container path such as /root/fwbase keeps the cell
#                  off the volume, e.g. while it is at quota: pull the results before the grace period ends)
#   --grace S      seconds between the end of the cell and the pod's stop (default 60; touch <out>/keep-pod to keep it)
#   --max-hours H  hard limit on the runner's wall time (default 8): a hung cell must not bill forever
set -uo pipefail
R=${R:-/workspace/oeh-campaign}
task=${1:?usage: cell.sh <task> <seed> [--out DIR] [--grace S] [--max-hours H]}; seed=${2:?seed}; shift 2
OUT=$R/fastestwam-baseline; grace=60; max_hours=8; stop=1
while [ $# -gt 0 ]; do
  case "$1" in
    --out) OUT=$2; shift 2;;
    --grace) grace=$2; shift 2;;
    --max-hours) max_hours=$2; shift 2;;
    --no-stop) stop=0; shift;;   # queue.sh runs many cells on one pod and stops it itself
    *) echo "unknown option $1" >&2; exit 2;;
  esac
done
ROBO=$R/RoboDojo
# What is evaluated comes from <out>/eval.env, written by launch.sh from the config; the defaults are the released
# OpenWAM-alpha RoboDojo checkpoint.
[ -f "$OUT/eval.env" ] && . "$OUT/eval.env"
CKPT_LABEL=${CKPT_LABEL:-OpenWAM-Alpha-Sim-RoboDojo}
POLICY_DIR=${POLICY_DIR:-XPolicyLab/policy/OpenWAM}; POLICY_ENV=${POLICY_ENV:-$R/envs/openwam}; policy=$(basename "$POLICY_DIR")
RUN_PREFIX=${RUN_PREFIX:-fwbase}; EVAL_NUM=${EVAL_NUM:-native}
export OPENWAM_CKPT_DIR=${CKPT_DIR:-$R/checkpoints/$CKPT_LABEL}   # the label alone goes into result paths, the weights come from here
cell=${task}__s${seed}; run_id=${RUN_PREFIX}_s${seed}
CELL=$OUT/cells/$cell; mkdir -p "$CELL" "$OUT/eval_result" "$OUT/smoke_results"
log() { echo "$(date -u +%FT%TZ) $*" | tee -a "$CELL/cell.log"; }

stop_pod() {  # the container's own pod id and pod-scoped key (a tmux shell does not inherit them; the key is never printed)
  local id key
  [ -e "$OUT/keep-pod" ] && { log "kept: $OUT/keep-pod exists"; return; }
  id=$(tr '\0' '\n' < /proc/1/environ | sed -n 's/^RUNPOD_POD_ID=//p')
  key=$(tr '\0' '\n' < /proc/1/environ | sed -n 's/^RUNPOD_API_KEY=//p')
  if [ -n "$id" ] && [ -n "$key" ]; then
    log "stopping this pod ($id)"
    curl -s -H "Content-Type: application/json" -H "Authorization: Bearer $key" https://api.runpod.io/graphql \
      -d "{\"query\":\"mutation { podStop(input:{podId:\\\"$id\\\"}) { id desiredStatus } }\"}"; echo
  else
    log "no pod id or pod key in this container: stop the pod from the Mac"
  fi
}
finish() {
  rc=$1; echo "$rc" > "$CELL/exit_code"
  [ "$stop" = 0 ] && { log "cell $cell exit $rc"; exit "$rc"; }
  log "cell $cell exit $rc; pod stops in ${grace}s"; sleep "$grace"; stop_pod; exit "$rc"
}

log "cell $cell on $(hostname): container-local setup"
[ -f "$OPENWAM_CKPT_DIR/config.yaml" ] || { log "no config.yaml in checkpoint dir $OPENWAM_CKPT_DIR"; finish 93; }
driver=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1 | tr -d ' ')
[ "${driver%%.*}" = 595 ] && { log "driver $driver crashes Isaac Sim"; finish 90; }
[ -f /etc/vulkan/icd.d/nvidia_icd.json ] || { log "no NVIDIA Vulkan ICD: Isaac Sim cannot render here"; finish 91; }
export DEBIAN_FRONTEND=noninteractive
if ! dpkg -s libglu1-mesa libvulkan1 libxt6 >/dev/null 2>&1; then
  (apt-get update -qq && apt-get install -y -qq --no-install-recommends git curl ca-certificates tmux jq ffmpeg rsync \
    libgl1 libglu1-mesa libglib2.0-0 libx11-6 libxext6 libxrender1 libsm6 libice6 libxrandr2 libxi6 libxcursor1 \
    libxinerama1 libegl1 libvulkan1 vulkan-tools libxt6 libexpat1) >> "$CELL/apt.log" 2>&1 || { log "apt failed"; finish 92; }
fi
# RoboDojo's scripts expect $HOME/miniconda3; caches (Isaac Sim shaders, kit) stay on the container disk.
[ -e "$HOME/miniconda3" ] || ln -s "$R/miniconda3" "$HOME/miniconda3"
export PATH=$R/miniconda3/bin:/usr/local/cuda/bin:$PATH
export OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y PRIVACY_CONSENT=Y PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1
git config --global --add safe.directory '*'

# Overlay root: every entry of the shared checkout by symlink, results redirected.
ROOT=/root/fwbase-root; mkdir -p "$ROOT"
for e in "$ROBO"/* "$ROBO"/.[!.]*; do
  n=$(basename "$e"); case "$n" in eval_result|smoke_results|.git) continue;; esac
  [ -e "$ROOT/$n" ] || ln -s "$e" "$ROOT/$n"
done
ln -sfn "$OUT/eval_result" "$ROOT/eval_result"       # -f: a pod reused with another --out must not keep the old target
ln -sfn "$OUT/smoke_results" "$ROOT/smoke_results"

python3 - "$CELL/meta.json" <<PY
import json, subprocess, sys, os, hashlib
def sh(c):
    return subprocess.run(c, shell=True, capture_output=True, text=True).stdout.strip()
ck = "$OPENWAM_CKPT_DIR"
json.dump({
  "cell": "$cell", "task": "$task", "seed": int("$seed"), "run_id": "$run_id", "out": "$OUT", "eval_num": "$EVAL_NUM",
  "policy_dir": "$POLICY_DIR", "policy_env": "$POLICY_ENV", "ckpt_label": "$CKPT_LABEL",
  "pod_id": sh("tr '\\\\0' '\\\\n' < /proc/1/environ | sed -n 's/^RUNPOD_POD_ID=//p'"), "hostname": sh("hostname"),
  "gpu": sh("nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader"),
  "robodojo_head": sh("git -C $ROBO rev-parse HEAD"), "robodojo_dirty": sh("git -C $ROBO status --porcelain --untracked-files=no | head -20"),
  "xpolicylab_head": sh("git -C $ROBO/XPolicyLab rev-parse HEAD"), "xpolicylab_dirty": sh("git -C $ROBO/XPolicyLab status --porcelain --untracked-files=no | head -20"),
  "deploy_yml": open("$ROBO/$POLICY_DIR/deploy.yml").read(),
  "sim_config": open("$ROBO/env_cfg/sim/sim_config.yml").read(),
  "ckpt_dir": ck, "ckpt_files": {f: os.path.getsize(os.path.join(ck, f)) for f in sorted(os.listdir(ck)) if os.path.isfile(os.path.join(ck, f))},
  "ckpt_observed": (open(ck + ".observed.json").read() if os.path.exists(ck + ".observed.json") else None),
  "started": sh("date -u +%FT%TZ"),
}, open(sys.argv[1], "w"), indent=1)
PY
grep -q "eval_batch: true" "$ROBO/$POLICY_DIR/deploy.yml" || { log "adapter eval_batch is not true"; finish 93; }

# GPU memory and utilisation every 10 s (the pilot's measurement; cheap enough to keep for every cell).
nvidia-smi --query-gpu=timestamp,memory.used,utilization.gpu --format=csv,noheader -l 10 > "$CELL/gpu.csv" 2>/dev/null &
smi=$!
log "runner start: task $task seed $seed run_id $run_id driver $driver out $OUT"
start=$(date +%s)
(cd "$ROOT" && timeout --signal=TERM --kill-after=120 "${max_hours}h" bash scripts/robodojo.sh benchmark \
  --policy-dir "$POLICY_DIR" --ckpt "$CKPT_LABEL" --policy-env "$POLICY_ENV" --eval-env RoboDojo \
  --only "$task" --eval-num "$EVAL_NUM" --seed "$seed" --gpu-ids 0 --run-id "$run_id" \
  --summary "$OUT/smoke_results/${run_id}__${task}.json" --markdown "$OUT/smoke_results/${run_id}__${task}.md") \
  > "$CELL/runner.log" 2>&1
rc=$?
kill $smi 2>/dev/null
wall=$(( $(date +%s) - start ))
res=$(ls "$OUT"/eval_result/RoboDojo/"$task"/"$policy"/arx_x5/"${seed}"_*/"${run_id}_${task}"/_result.json 2>/dev/null | head -1)
peak=$(awk -F', ' '{gsub(/ MiB/,"",$2); if ($2+0>m) m=$2+0} END {print m+0}' "$CELL/gpu.csv" 2>/dev/null)
python3 - "$CELL/meta.json" "$rc" "$wall" "${res:-}" "${peak:-0}" <<'PY'
import json, sys
p, rc, wall, res, peak = sys.argv[1:]
m = json.load(open(p)); m.update(runner_exit=int(rc), wall_seconds=int(wall), result_json=res or None, peak_gpu_mib=int(float(peak)))
if res:
    r = json.load(open(res)); m.update(success_rate=r.get("success_rate"), score=r.get("score"), eval_time=r.get("eval_time"))
json.dump(m, open(p, "w"), indent=1)
print({k: m.get(k) for k in ("cell", "runner_exit", "wall_seconds", "peak_gpu_mib", "success_rate", "score", "eval_time")})
PY
[ -n "$res" ] || { log "no _result.json (runner exit $rc, ${wall}s)"; [ "$rc" = 0 ] && rc=94; }
finish "$rc"
