# Fastest-WAM work records

Records of the Fastest-WAM training and evaluation line built on this OpenWAM fork.
They describe runs on the 8×H100 training server; raw results, logs and checkpoints stay there.

| File | Content |
|---|---|
| [STATE-2026-10-02.md](STATE-2026-10-02.md) | Where the work lives, every run and its state, decisions to carry forward, next experiments |
| [checkpoints.md](checkpoints.md) | Checkpoint paths, sizes and SHA-256 on the training server |
| [training/robodojo-resumable-training.md](training/robodojo-resumable-training.md) | Resumable RoboDojo training protocol and the paused 7-GPU FM run |
| [training/robodojo-video-xm8-20260928.md](training/robodojo-video-xm8-20260928.md) | 8-GPU video Forward XM run and the fair-comparison contract |
| [evaluation/libero-paired-2026-09-26/](evaluation/libero-paired-2026-09-26/README.md) | Paired FM-2 vs mixed-MIP LIBERO evaluation, Fast-WAM readout diagnostics, LIBERO-Plus screens, with JSON audits |
| [evaluation/libero-plus-baseline/](evaluation/libero-plus-baseline/README.md) | LIBERO-Plus public baseline intake: sources, pinned revisions, download helpers |

## Code branches

Stacked in this order, each one commit on top of the previous, starting from `main`:

1. `evan/robodojo-dataloader`: RoboDojo dataloader path
2. `evan/mip-objective`: MIP action objective beside flow matching
3. `evan/resumable-checkpointing`: early saves, full resume state kept at completion
4. `evan/video-forward-xm`: optional Forward XM (K=2) on the video loss
5. `evan/experiment-scripts`: training launchers and LIBERO diagnostic scripts

Config paths and launch scripts are specific to the training server.
