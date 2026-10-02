# RoboDojo eval on RunPod: full protocol and single tasks

Config files plus a launcher for RoboDojo's own runner (`scripts/robodojo.sh benchmark`, native episode counts,
adapter `eval_batch: true`) on L40S pods of the `oeh-campaign-eu` volume. Built to reproduce the OpenWAM-α leaderboard
entry (Avg SR 11.92 / Score 17.18) and to evaluate our own checkpoints the same way later.

**Status (2026-10-02): the full eval has not been run.** One pilot cell finished and four more were started and
stopped; see "What exists on the volume".

## Use

```bash
bash launch.sh configs/openwam-alpha_single.env TASKS=stack_bowls        # prints the plan, starts nothing
bash launch.sh configs/openwam-alpha_single.env TASKS=stack_bowls --go   # one task, seed 0, 1 pod
bash launch.sh configs/openwam-alpha_full.env --go                       # 42 tasks x 3 seeds, 24 pods
bash status.sh configs/openwam-alpha_full.env                            # cells, pods, balance
bash collect.sh configs/openwam-alpha_full.env                           # pull records, compare with the reference
```

`--go` spends money (about $1.09 per pod-hour). Any config key can be overridden on the command line
(`TASKS=memory SEEDS=0 PODS=6`). Running `launch.sh ... --go` again adds pods to the same eval.

| Config | Cells | Pods | Estimate |
|---|---|---|---|
| `configs/openwam-alpha_full.env` | 54 task dirs × seeds 0,1,2 = 162 | 24 | ~196 L40S-hours, ~$215, ~8 h |
| `configs/openwam-alpha_single.env` | 1 task (2 dirs for a Generalization task) × seed 0 | 1 | ~1–4 h, ~$1–5 |
| `configs/custom-ckpt_template.env` | template for our own checkpoints | 1 | as above |

The estimate is RoboDojo's own per-task runtime table (`../tasks.json`) times 1.39, the factor the pilot measured on an
L40S. One measurement: treat it as ±15%.

## How it runs

- A **cell** is one (task dir, seed). `plan.py` turns a config into a queue of cells, longest first.
- Each pod runs `pod/queue.sh`: claim the next free cell (`mkdir <OUT>/claims/<cell>`), run it with `pod/cell.sh`,
  repeat, and stop the pod when the queue is empty. Pods can join at any time.
- Pod count sets the wall time only: cost is ~flat (about 3 min of boot per pod). The floor is the longest cell,
  ~4.1 h (`imitate_sorting_sequence`). One cell needs 39.9 GB of the L40S's 46 GB, so one cell per GPU.
- Every task pays ~10 min of start-up (checkpoint load from the volume, then Isaac Sim).
- **Spend guard:** `<OUT>/claim_limit` caps the cells that may be claimed. A pod that finds the cap reached stops
  itself. `CLAIM_LIMIT=all` allows every cell of the launch; a number allows that many new cells.
- **Shared checkouts stay untouched:** the runner starts from `/root/fwbase-root`, a directory of symlinks to
  `$R/RoboDojo` on the pod's own disk, with `eval_result/` and `smoke_results/` pointing into `OUT`.
- **Drivers:** RunPod cannot pin one. `DRIVERS` lists what is accepted; a pod on another driver is deleted unused.
  On 2026-10-02 only 3 of 11 draws were 580.159.04 (7 were 580.173.02), so one driver means a small fleet.
  The driver is recorded per cell in `meta.json`.
- Each cell writes `<OUT>/cells/<cell>/meta.json` (commits, driver, checkpoint files, wall time, peak GPU memory,
  success rate, score) next to RoboDojo's own `_result.json` and per-episode videos (~215 MB per 50 episodes).

## Evaluating another checkpoint

Copy `configs/custom-ckpt_template.env`, set `NAME`, `OUT`, `CKPT_DIR`, `CKPT_LABEL`. The checkpoint directory must
already be on the volume with `config.yaml`, `checkpoint_step_*.safetensors` and `normalization_stats.npy`. Use a new
`OUT` per checkpoint: `launch.sh` refuses an `OUT` that holds another checkpoint's results.

## What exists on the volume

`/workspace/oeh-campaign/fastestwam-baseline/` (RoboDojo ee67a14, XPolicyLab fa431ec, checkpoint rev 2c13022,
driver 580.159.04):

- `put_bottles_into_dustbin` seed 0, finished: SR 96 / Score 96.8 (reference seed 0: 98 / 98.0), 67 min.
- `imitate_sorting_sequence` seeds 0–2 and `pour_by_language` seed 2: stopped after 15–25 min, no result. Their
  claims are released by the next `launch.sh --go`. RoboDojo keeps a resume manifest per run id, so they may resume
  rather than restart; that path has not been exercised here.
- `claim_limit` is 0, so nothing can run there until a launch sets it.

## Not yet exercised

`pod/cell.sh` and `pod/queue.sh` ran on four pods before they gained config reading (`eval.env`, `--queue`).
`launch.sh --go`, `status.sh` and `collect.sh` in their config-driven form have not been run against a pod: start with
the single-task config. `../compare_to_reference.py` reproduces the published aggregates when fed the reference itself.

## Files

- `configs/*.env`: what to evaluate. `../tasks.json`: the 54 task dirs, dimensions, paired `_random` siblings, runtimes.
- `launch.sh`, `status.sh`, `collect.sh`, `plan.py`, `_lib.sh`: operator side. They need `OEH_REPO` set to an
  OpenEmbodied-Harness checkout, whose RunPod helpers and account settings (`~/.runpod/config.toml`,
  `~/.config/runpod-eval/env`) they use.
- `../compare_to_reference.py`: per-task and per-dimension comparison with the reference.
- `pod/cell.sh`, `pod/queue.sh`: uploaded to `<OUT>/bin/` by each launch.
- `assets/robodojo_verification/openwam_robodojo_per_seed.json` (repo root): the per-seed reference.
- `results/<NAME>/`: collected records. `results/openwam-alpha/` holds the pilot cell.
