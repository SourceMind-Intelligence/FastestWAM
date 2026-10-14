# LIBERO DoT + SF MIP-2 vs DoT + FM

## Why DoT

Deep 30L MoT ActionDiT already makes single/few-step FM near-ceiling on LIBERO,
so detached Self-Forcing MIP-2 has little room. **DoT** (Denoising Transformer)
is a shallow ActionDiT that cross-attends a few video bridges — iteration / SF
is more likely to help when each action forward is weak.

## Definition (this repo)

```text
model=dual_system_dot
  variant = joint_cross_attn
  dot_num_layers = 4          # bridges ≈ (0, 10, 19, 29) on Wan-30
  Video FM unchanged
```

Warm-start from OpenWAM-Alpha MoT foundation with
`training.finetune_strict_load=false` (video weights transfer; DoT action mostly
re-init where names/shapes differ).

## Runs

| ID | Name | Objective |
|----|------|-----------|
| SF | `SF_mip_mixed` | MIP-2 Mixed Self-Forcing |
| FM | `FM_flow` | OpenWAM flow-matching |

Inference: SF = 2 NFE MIP; FM compare at **2 denoise steps**.

## Budget (≤2 days)

- **8×H100**, sequential: SF first, then FM
- `num_epochs=4` (override with `NUM_EPOCHS=`)
- global batch 128 via `8 × bs=8 × accum=2`
- Prior MoT FM 5-ep ≈16h on 4GPU; expect SF≈10–14h, FM≈6–9h on 8GPU DoT

## Disk policy (tight ≈1–2TiB free)

Each model-only ckpt ≈ **22GiB**; full DeepSpeed resume ≈ **90GiB**.

| Knob | Value |
|------|-------|
| `save_full_states_for_resume` | **false** (no accel_state) |
| `keep_last_k_ckpts` | 1 |
| `max_milestone_ckpts` | 1 |
| `milestone_save_steps` | `[4000]` only |
| `disk_guard_critical_gb` | 400 |

Peak per run ≈ **≤45GiB** (1 rolling + 1 milestone); final keeps 1 weights file.

## Launch

```bash
# smoke
bash scripts/experiments/run_libero_dot_sf_suite.sh smoke

# train SF first (8 GPUs)
WAIT_FOR_SUITE=0 bash scripts/experiments/run_libero_dot_sf_suite.sh train_sf

# after SF finishes
WAIT_FOR_SUITE=0 bash scripts/experiments/run_libero_dot_sf_suite.sh train_fm

# or both sequential
bash scripts/experiments/run_libero_dot_sf_suite.sh overnight
```
