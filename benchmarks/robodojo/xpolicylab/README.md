# RoboDojo through XPolicyLab with history frames

XPolicyLab's OpenWAM adapter (`policy/OpenWAM/model.py`, XPolicyLab fa431ec) conditions every action chunk on the
current frame only, and its vendored OpenWAM has no history code. A checkpoint trained with
`dataloader.history_num_frames` (R4: the episode's first frame plus the frames 6, 4 and 2 s back) needs the past
frames at evaluation too. This directory serves such checkpoints through RoboDojo's own runner without changing it.

| File | What it does |
|---|---|
| `history_model.py` | Subclass of the stock adapter. One `FrameHistory` per parallel env, fed the composed canvas on every `update_obs_batch` (XPolicyLab calls it once per env step) and cleared on `reset`; chunks are generated env by env through this repository's `engine.generate`. Everything else is the stock adapter's. |
| `stage_xpolicylab.sh` | Pod side. Builds a container-local copy of XPolicyLab's code with `history_model.py` as `policy/OpenWAM/model.py` (the stock file beside it as `openwam_stock_model.py`, sha256-checked) and `policy/OpenWAM/OpenWAM` linked to a FastestWAM checkout, then points the RunPod kit's run root (`/root/fwbase-root/XPolicyLab`) at it. The shared checkout on the volume is not written. |
| `adapter_actions.py` | Drives an adapter through synthetic episodes (no Isaac Sim) in XPolicyLab's batch-loop order, saves the actions and timings, and compares two runs. |

## Why it stays correct

- One env step is 0.04 s: `collect_freq` 25 and `dt` 0.004 give 10 physics steps per action, the rate of the training
  data (`additional_info/frequency` 25), so `history_stride` 50 is 2 s in both.
- XPolicyLab's loop (`policy/OpenWAM/deploy.py`, `eval_one_episode_batch`) calls `update_obs_batch` before each chunk
  and after every non-final action, for the envs still running, so every observation reaches the adapter exactly once.
- `FrameHistory` (openwam/deploy/policy.py) is the sampler the WebSocket server uses; the frame indices match the
  training reader's (`openwam/dataloader/robodojo.py`). Tests: `tests/test_xpolicylab_history_adapter.py`.
- A checkpoint without history goes through the same code with no history, one env per `generate` call. Comparing
  it with the stock adapter on the same weights (below) isolates the tree and batching difference from the history.

## Use on a pod

```bash
git clone https://github.com/SourceMind-Intelligence/FastestWAM /root/fastestwam
git -C /root/fastestwam checkout <commit>
bash /root/fastestwam/benchmarks/robodojo/xpolicylab/stage_xpolicylab.sh /root/fastestwam \
    --python /workspace/oeh-campaign/envs/openwam/bin/python
# then the kit as usual (POLICY_DIR stays XPolicyLab/policy/OpenWAM); result paths keep the policy name OpenWAM,
# so runs are told apart by the checkpoint label and /root/xpl-history/stage.json.
```

Before a panel, on the same machine:

```bash
cd /tmp && export PYTHONPATH=/root/fwbase-root
PY=/workspace/oeh-campaign/envs/openwam/bin/python; A=/root/fastestwam/benchmarks/robodojo/xpolicylab/adapter_actions.py
VENDORED=/workspace/oeh-campaign/RoboDojo/XPolicyLab/policy/OpenWAM/OpenWAM
$PY $A --module XPolicyLab.policy.OpenWAM.openwam_stock_model --openwam-root $VENDORED --ckpt-dir <alpha> --out stock.npz
$PY $A --openwam-root /root/fastestwam --ckpt-dir <alpha> --out ours.npz
$PY $A --compare stock.npz ours.npz             # same weights: tree and batching differences only
$PY $A --openwam-root /root/fastestwam --ckpt-dir <history ckpt> --steps 224 --out history.npz   # timing with history
```
