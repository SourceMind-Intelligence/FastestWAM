#!/usr/bin/env bash
# Mac side. Pull the records of an eval (every _result.json, each cell's meta.json and logs, the runner summaries; no
# videos) to results/<NAME>/ and print the comparison with the reference.
#   collect.sh <config.env> [pod id]     through a running pod of this eval, or any running pod on the volume
set -uo pipefail
source "$(dirname "$0")/_lib.sh"
load_config "${1:?usage: collect.sh <config.env> [pod id]}"
id=${2:-$(eval_pods | head -1)}
[ -n "$id" ] || { echo "no running pod to read the volume through: pass the id of any running pod on the volume"; exit 3; }
read -r h p < <(pod target "$id") || exit 1
dest=$HERE/results/$NAME; mkdir -p "$dest"
rsync -az --no-owner --no-group -e "ssh -o StrictHostKeyChecking=accept-new -p $p" --prune-empty-dirs \
  --include '*/' --include '_result.json' --include 'meta.json' --include 'exit_code' --include 'cell.log' \
  --include '*.md' --include 'eval.env' --include 'owner' --exclude '*' "root@$h:$OUT/" "$dest/"
echo "pulled to $dest: $(find "$dest" -name _result.json | wc -l | tr -d ' ') _result.json, $(find "$dest/cells" -name meta.json | wc -l | tr -d ' ') cell records"
python3 "$HERE/../compare_to_reference.py" "$dest"
