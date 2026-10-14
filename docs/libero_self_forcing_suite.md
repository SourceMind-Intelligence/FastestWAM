# LIBERO Self-Forcing MIP-2 Suite

## Hypothesis

Original MIP-2 fails on LIBERO precision tasks (esp. task06) mainly because
**refinement training distribution ≠ inference prediction-error distribution**,
not because 2-step iterative computation is insufficient.

Core experiment: **Mixed Self-Forcing MIP-2** (`0.5 GT + 0.5 pred0.detach()`).

## Runs

| ID | Name | Action objective | Refine input |
|----|------|------------------|--------------|
| A | `A_fm` | OpenWAM FM | native |
| B | `B_mip_gt` | MIP-2 | `0.9*GT + 0.1*z` |
| C | `C_mip_mixed` | MIP-2 mixed SF | `0.9*(0.5 GT + 0.5 pred0.detach()) + 0.1*z` |
| D | `D_mip_pred` | MIP-2 pred SF | `0.9*pred0.detach() + 0.1*z` |

Inference for B/C/D is identical: zeros → pred0 → `0.9*pred0` → pred1.

## Hardware layout (revised)

**2×H100 cannot hold DualSystem + ZeRO-2 + MIP-2** at useful batch (OOM on
optimizer / second MIP branch). Suite uses:

```text
Wave 1:  C on GPU 0-3   |   A on GPU 4-7
Wave 2:  B on GPU 0-3   |   D on GPU 4-7
```

- Per run: 4 GPUs, `batch_size=8`, `grad_accum=4` → **global batch 128**
- Budget: official `num_epochs=5`
- Milestones (opt-step): 4000 / 8000 / 10000 model-only
- Full resume: `max_full_ckpts=1`

## Launch

```bash
# smoke Mixed SF on 4 GPUs
bash scripts/experiments/run_libero_self_forcing_suite.sh smoke

# wave 1 (C + A)
WAIT_FOR_SUITE=0 bash scripts/experiments/run_libero_self_forcing_suite.sh wave1

# after wave 1 finishes
WAIT_FOR_SUITE=0 bash scripts/experiments/run_libero_self_forcing_suite.sh wave2

# FLOPs probe (batch=1, measure_flops=true)
bash scripts/experiments/probe_step_flops.sh mip mixed
bash scripts/experiments/probe_step_flops.sh flow
```

## Config knobs

- `model.architecture.mip_refine_mode`: `gt | mixed | pred`
- `training.milestone_save_steps`, `max_full_ckpts`, disk guards
- `training.measure_flops` (off by default — FlopCounterMode spikes VRAM)

## Success criteria

1. Mixed SF ≫ Original MIP on **task06**
2. Overall SR does not drop
3. Same-TFLOPs comparison vs FM (report both same-step and same-FLOPs)
