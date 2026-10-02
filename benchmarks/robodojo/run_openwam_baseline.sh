#!/bin/bash
# Sweep the released OpenWAM-Alpha-Sim-RoboDojo checkpoint over all 42 RoboDojo
# sim tasks (54 task dirs: the 12 Generalization tasks also run their _random
# sibling) x seeds 0 1 2, through XPolicyLab's OpenWAM adapter. Then compare:
#
#   python benchmarks/robodojo/compare_to_reference.py <RoboDojo>/eval_result/RoboDojo
#
# Run from anywhere on a GPU host with RoboDojo + XPolicyLab installed, e.g.
#
#   XPL_OPENWAM_DIR=/path/to/RoboDojo/XPolicyLab/policy/OpenWAM \
#   POLICY_ENV=openwam EVAL_ENV=robodojo POLICY_GPU=0 ENV_GPU=1 \
#   bash benchmarks/robodojo/run_openwam_baseline.sh
#
# Optional: CKPT (default OpenWAM-Alpha-Sim-RoboDojo, resolved under the
# adapter's checkpoints/), SEEDS (default "0 1 2"), TASKS (space-separated
# subset). Failed cells are logged and skipped; rerunning only adds a newer
# timestamp, and the comparison reads the latest one per seed.
set -uo pipefail

: "${XPL_OPENWAM_DIR:?set XPL_OPENWAM_DIR to XPolicyLab/policy/OpenWAM}"
: "${POLICY_ENV:?set POLICY_ENV to the policy conda env or uv path}"
: "${EVAL_ENV:?set EVAL_ENV to the RoboDojo eval conda env}"
CKPT=${CKPT:-OpenWAM-Alpha-Sim-RoboDojo}
SEEDS=${SEEDS:-"0 1 2"}
POLICY_GPU=${POLICY_GPU:-0}
ENV_GPU=${ENV_GPU:-0}

GEN_TASKS="stack_bowls push_T pack_objects_into_box fold_clothes hang_mugs sweep_blocks
pour_liquid_into_cup make_toast arrange_largest_number sort_nesting_dolls_by_size
store_laptop_and_headphones stack_blocks"
OTHER_TASKS="fasten_screws plug_in_charger insert_tubes pour_balls_into_vase play_Xylophone
deposit_coin insert_key build_tower
put_bottles_into_dustbin fill_pen_holder classify_objects play_tic_tac_toe fill_egg_holder
organize_table make_kong play_stacking_toy
cover_blocks match_and_pick_from_conveyor swap_blocks swap_T press_by_number imitate_sorting_sequence
align_blocks general_pickup stack_blocks_by_language solve_equation classify_objects_by_language
pick_from_conveyor_by_image store_tools_in_toolbox pour_by_language"

if [[ -z "${TASKS:-}" ]]; then
    TASKS="${OTHER_TASKS}"
    for t in ${GEN_TASKS}; do TASKS="${TASKS} ${t} ${t}_random"; done
fi

LOG_DIR=${LOG_DIR:-"${XPL_OPENWAM_DIR}/logs/robodojo_baseline_$(date +%Y%m%d_%H%M%S)"}
mkdir -p "${LOG_DIR}"
failed=()
for seed in ${SEEDS}; do
    for task in ${TASKS}; do
        log="${LOG_DIR}/${task}_seed${seed}.log"
        echo "[$(date +%H:%M:%S)] ${task} seed=${seed} -> ${log}"
        if ! (cd "${XPL_OPENWAM_DIR}" && bash eval.sh RoboDojo "${task}" "${CKPT}" arx_x5 ee \
                "${seed}" "${POLICY_GPU}" "${ENV_GPU}" "${POLICY_ENV}" "${EVAL_ENV}") >"${log}" 2>&1; then
            echo "  FAILED (see log)"
            failed+=("${task}:${seed}")
        fi
    done
done

echo "Done. Failed cells: ${#failed[@]} ${failed[*]:-}"
[[ ${#failed[@]} -eq 0 ]]
