#!/usr/bin/env bash
# Mac side. One line per claimed cell (owner pod, exit code, episodes so far, last GPU sample), counts and balance.
#   status.sh <config.env> [pod id]      reads the volume through a running pod of this eval, or the pod given
set -uo pipefail
source "$(dirname "$0")/_lib.sh"
load_config "${1:?usage: status.sh <config.env> [pod id]}"
pods=$(eval_pods)
echo "running pods of $NAME: $(echo $pods | wc -w | tr -d ' ')  ($(echo $pods))"
runpodctl user 2>/dev/null | grep -E "clientBalance|currentSpendPerHr" | tr -d '\n'; echo
id=${2:-$(echo $pods | awk '{print $1}')}
[ -n "$id" ] || { echo "no running pod to read the volume through: pass the id of any running pod on the volume"; exit 3; }
pod_run "$id" "O=$OUT; . \$O/eval.env"'
for c in $O/claims/*/; do n=$(basename $c); t=${n%__s*}; s=${n##*__s}; L=$O/smoke_results/${RUN_PREFIX}_s$s/logs/$t.log
  printf "%-44s %-15s exit=%-3s %s | %s\n" "$n" "$(cut -d" " -f1 $c/owner)" "$(cat $O/cells/$n/exit_code 2>/dev/null)" \
    "$(grep -a "Success nums" $L 2>/dev/null | tail -1)" "$(tail -1 $O/cells/$n/gpu.csv 2>/dev/null | cut -d, -f2-)"
done
echo "claimed $(ls $O/claims | wc -l) / limit $(cat $O/claim_limit); finished $(ls $O/cells/*/exit_code 2>/dev/null | wc -l); failed: $(ls $O/failed | tr "\n" " ")"
echo "results: $(find $O/eval_result -name _result.json | wc -l) _result.json, $(du -sh $O/eval_result | cut -f1)"'
