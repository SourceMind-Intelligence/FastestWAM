# Sourced by launch.sh, status.sh and collect.sh: the OEH RunPod helpers (pod, pod_run, pod_pipe) and config reading.
# The RunPod account settings stay where the OEH tooling keeps them (~/.runpod/config.toml, ~/.config/runpod-eval/env).
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# OEH_REPO: a checkout of OpenEmbodied-Harness, whose runpod-eval scripts create pods and reach them over ssh.
[ -f "${OEH_REPO:-}/.agents/skills/runpod-eval/scripts/_common.sh" ] || { echo "set OEH_REPO to an OpenEmbodied-Harness checkout (it provides the RunPod pod helpers)" >&2; exit 2; }
export OEH_REPO
source "$OEH_REPO/.agents/skills/runpod-eval/scripts/_common.sh"
log() { echo "$(date -u +%H:%M:%SZ) $*"; }
# cfg <config.env> [KEY=VALUE ...]: the config's keys as shell variables, overrides last
load_config() {
  local c=${1:?config}; shift
  [ -f "$c" ] || { echo "no such config: $c" >&2; exit 2; }
  PAIR_RANDOM=1; PODS=1; DRIVERS="580.159.04"; CLAIM_LIMIT=all; POLICY_DIR=XPolicyLab/policy/OpenWAM
  set -a; . "$c"; set +a
  local kv; for kv in "$@"; do export "${kv%%=*}=${kv#*=}"; done
}
# Running pods of this eval (their names carry -fw-<NAME>-), then any running pod on the volume.
eval_pods() { pod list | awk -v l="-fw-$NAME-" '$2 == "RUNNING" && index($0, l) { print $1 }'; }
