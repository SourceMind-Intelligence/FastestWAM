# Checkpoint manifest

Checkpoints stay on the training server under
`/root/evan/Fastest-WAM-evan/outputs/openwam_checkpoints/`. Each weights file is
24,813,767,464 bytes. SHA-256 computed on the server on 2026-10-02.

| Checkpoint | Path (relative to the directory above) | Resume state | SHA-256 |
|---|---|---|---|
| RoboDojo pilot, step 4,000 | `pilot_step4000/checkpoint_step_4000.safetensors` | no | `dc49c9ecabc55a4932819e5363bd49e8d7ee3f1ec4e566a518a47ce4906adfcf` |
| RoboDojo pilot, step 10,000 | `pilot_step10000/checkpoint_step_10000.safetensors` | no | `ad9a344856ad6b9e456a72359988c0b088aee9e42ae19dd07e1ae318ea337248` |
| RoboDojo pilot, step 24,000 (warm start for both runs below) | `2026-09-17_15-32-18/checkpoint_step_24000.safetensors` | no | `f0f0d99f509ed44882d3d08107587f88e26f0863eb9d41b182e42ce868c4f705` |
| FM, 7 GPU, paused at step 37,000 | `robodojo_5epoch_resumable_20260928/2026-09-28_00-51-21/checkpoint_step_37000.safetensors` | `accel_state_step_37000/` (also step 36,000) | `d36a43094a52f0391bada8e77fce8f084cb0ec996d0e296b7514bba7e15716aa` |
| Video Forward XM, 8 GPU, final step 72,505 | `robodojo_video_xm8_20260928/2026-09-28_23-50-17/checkpoint_step_72505.safetensors` | `accel_state_step_72505/` (also step 72,000) | `dd0e895bd15a3e73ecaeebdf71f46563ab42cd9ad52c76fe0c09f054b6d73f4c` |

Not hashed: the step 36,000 and 72,000 weights, the step-20 debug checkpoint, and the
`robodojo_resume_smoke_20260928` pre-flight (steps 2 and 3).

The XM run's `run_provenance/` hashes for `base.py`, `openwam_trainer.py`,
`training_utils.py`, `configs/train.yaml` and the FM control launcher match the files on
branch `evan/experiment-scripts`, so that branch is the code that produced the XM checkpoint.

The paired LIBERO checkpoints (A FM and C mixed-MIP, step 42,730) belong to Tengfei's
checkout under `/root/tengfei/FastestWAM/outputs/experiments/libero_sf_suite_2026-09-25_00-52-16/`.
