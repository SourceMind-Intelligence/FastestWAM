#!/usr/bin/env bash
# Pod side, before the RunPod kit's queue.sh / cell.sh (benchmarks/robodojo/runpod on evan/stable): serve checkpoints
# through history_model.py (this directory) instead of XPolicyLab's stock OpenWAM adapter, without writing into the
# shared RoboDojo checkout on the volume.
#   stage_xpolicylab.sh <fastestwam-checkout> [--robodojo DIR] [--run-root DIR] [--copy DIR] [--python PY]
#
# Builds, on the container disk:
#   <copy>/XPolicyLab  a copy of <robodojo>/XPolicyLab's code (no .git, caches or checkpoints; other policy dirs and
#                      any other directory are linked, not copied). In policy/OpenWAM, model.py is this directory's
#                      history_model.py, the stock model.py sits beside it as openwam_stock_model.py (its sha256 must
#                      be the version the adapter was written against), and OpenWAM/ links to <fastestwam-checkout>
#                      in place of the vendored copy, so setup_eval_policy_server.sh's default OPENWAM_ROOT is ours.
#   <copy>/<entry>     a link to every other entry of <robodojo> except .git, so XPolicyLab code that reads
#                      ../../env_cfg relative to itself still finds it.
#   <run-root>/XPolicyLab -> <copy>/XPolicyLab. cell.sh links the run root's other entries itself and keeps an
#                      existing one, so only the policy server changes: setup_eval_policy_server.sh runs
#                      setup_policy_server.py from the copy, Python puts that script's real directory first on
#                      sys.path, and XPolicyLab.policy.OpenWAM.model resolves to the history adapter. The client
#                      (Isaac Sim side) uses the same deploy.py and deploy.yml as before, whichever checkout it imports.
# Result paths keep the policy name OpenWAM: tell runs apart by the checkpoint label and <copy>/stage.json.
# With --python (the policy env's python), the adapter is imported the way the server imports it, without loading
# weights, and the files it resolved to are printed.
set -euo pipefail
STOCK_SHA256=70f3699eddbea45b01360cf4e7bc5a6c4ff9672e3cb38e6fb7d2da2b5d9220f4   # XPolicyLab fa431ec policy/OpenWAM/model.py

src=""; robo=/workspace/oeh-campaign/RoboDojo; run_root=/root/fwbase-root; copy=/root/xpl-history; py=""
while [ $# -gt 0 ]; do
  case "$1" in
    --robodojo) robo=$2; shift 2;;
    --run-root) run_root=$2; shift 2;;
    --copy) copy=$2; shift 2;;
    --python) py=$2; shift 2;;
    -*) echo "unknown option $1" >&2; exit 2;;
    *) src=$1; shift;;
  esac
done
[ -n "$src" ] || { echo "usage: stage_xpolicylab.sh <fastestwam-checkout> [--robodojo DIR] [--run-root DIR] [--copy DIR] [--python PY]" >&2; exit 2; }
src=$(cd "$src" && pwd); robo=$(cd "$robo" && pwd)
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
xpl=$robo/XPolicyLab
stock=$xpl/policy/OpenWAM/model.py

grep -q "^class FrameHistory" "$src/openwam/deploy/policy.py" || { echo "$src has no openwam.deploy.policy.FrameHistory" >&2; exit 1; }
[ -f "$stock" ] || { echo "no stock adapter at $stock" >&2; exit 1; }
sha=$(sha256sum "$stock" | cut -d' ' -f1)
[ "$sha" = "$STOCK_SHA256" ] || { echo "stock adapter sha256 $sha, expected $STOCK_SHA256 (XPolicyLab fa431ec)" >&2; exit 1; }
case "$copy" in /root/?*) ;; *) echo "--copy must be a directory under /root (it is deleted and rebuilt)" >&2; exit 2;; esac
if [ -e "$run_root/XPolicyLab" ] && [ ! -L "$run_root/XPolicyLab" ]; then
  echo "$run_root/XPolicyLab exists and is not a link; refusing to replace it" >&2; exit 1
fi

entries() {  # every entry of a directory, dotfiles included, one per line
  local d=$1 e
  for e in "$d"/* "$d"/.[!.]* "$d"/..?*; do if [ -e "$e" ] || [ -L "$e" ]; then echo "$e"; fi; done
}
rm -rf "$copy"; mkdir -p "$copy/XPolicyLab/policy/OpenWAM"
while IFS= read -r e; do
  n=$(basename "$e"); case "$n" in XPolicyLab|.git) continue;; esac
  ln -s "$e" "$copy/$n"
done < <(entries "$robo")
while IFS= read -r e; do
  n=$(basename "$e")
  case "$n" in
    .git|__pycache__|policy) continue;;
    client_server|utils|scripts) cp -a "$e" "$copy/XPolicyLab/$n";;
    *) if [ -f "$e" ]; then cp -p "$e" "$copy/XPolicyLab/$n"; else ln -s "$e" "$copy/XPolicyLab/$n"; fi;;
  esac
done < <(entries "$xpl")
while IFS= read -r e; do
  n=$(basename "$e")
  case "$n" in
    OpenWAM|__pycache__) continue;;
    *) if [ -f "$e" ]; then cp -p "$e" "$copy/XPolicyLab/policy/$n"; else ln -s "$e" "$copy/XPolicyLab/policy/$n"; fi;;
  esac
done < <(entries "$xpl/policy")
while IFS= read -r e; do
  n=$(basename "$e")
  case "$n" in
    OpenWAM|checkpoints|__pycache__|model.py) continue;;
    *) if [ -f "$e" ]; then cp -p "$e" "$copy/XPolicyLab/policy/OpenWAM/$n"; else ln -s "$e" "$copy/XPolicyLab/policy/OpenWAM/$n"; fi;;
  esac
done < <(entries "$xpl/policy/OpenWAM")
find "$copy/XPolicyLab" -name __pycache__ -type d -prune -exec rm -rf {} +
pol=$copy/XPolicyLab/policy/OpenWAM
cp -p "$stock" "$pol/openwam_stock_model.py"
cp "$here/history_model.py" "$pol/model.py"
ln -s "$src" "$pol/OpenWAM"
mkdir -p "$run_root"
ln -sfn "$copy/XPolicyLab" "$run_root/XPolicyLab"

commit=$(git -C "$src" rev-parse HEAD 2>/dev/null || echo unknown)
python3 - "$copy/stage.json" "$src" "$commit" "$robo" "$sha" "$pol/model.py" "$run_root" <<'PY'
import hashlib, json, sys, time
out, src, commit, robo, stock_sha, adapter, run_root = sys.argv[1:]
json.dump({"staged_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "fastestwam": src, "fastestwam_commit": commit,
           "robodojo": robo, "stock_model_sha256": stock_sha,
           "adapter_sha256": hashlib.sha256(open(adapter, "rb").read()).hexdigest(), "run_root": run_root},
          open(out, "w"), indent=1)
PY
echo "staged: $run_root/XPolicyLab -> $copy/XPolicyLab; openwam from $src ($commit)"

if [ -n "$py" ]; then
  # Same resolution as the server: the script runs from the copy, so its real directory leads sys.path.
  cat > "$copy/XPolicyLab/_stage_check.py" <<'PY'
import importlib, sys
m = importlib.import_module("XPolicyLab.policy.OpenWAM.model")
base = sys.modules[m.OpenWAMModel.__module__]
pd = importlib.import_module("XPolicyLab.utils.process_data")
print("adapter     ", m.__file__)
print("stock base  ", base.__file__)
print("process_data", pd.__file__)
assert m.__file__.startswith(sys.argv[1]), "the adapter did not come from the copy"
PY
  (cd / && PYTHONPATH="$run_root:$src${PYTHONPATH:+:$PYTHONPATH}" "$py" "$run_root/XPolicyLab/_stage_check.py" "$copy/XPolicyLab")
  rm -f "$copy/XPolicyLab/_stage_check.py"
fi
