# LIBERO DoT Results: Architecture × Objective Analysis

**Updated:** 2026-10-18 (raw-result audit) · **Branch:** `tengfei`

## Status

| Job | Status |
|-----|--------|
| DoT-FM train (4ep) | done → 94.15% |
| DoT-SF Mixed from foundation | done → **4.55%** |
| Fair FM continue +2ep (freeze) | done → **94.20%** |
| Fair SF Mixed from FM +2ep (freeze) | done → **3.20%** |
| FM→MIP-gt soft convert +2ep | done → **2.90%** |
| DoT-8 FM / DoT-4 MIP-gt from foundation | done → **93.90% / 3.30%** |

---

## 1. Main result table (LIBERO 40×50, sync, prefer 2-step)

| Rank | Model | Arch | Action depth | Objective | Overall | spatial | object | goal | 10 |
|-----:|-------|------|-------------:|-----------|--------:|--------:|-------:|-----:|---:|
| 1 | Alpha sim-libero | MoT | ≈30 | FM | **99.55%** | 99.4 | 99.8 | 99.6 | 99.4 |
| 2 | MoT-SF Mixed MIP-2 | MoT | ≈30 | mip mixed | **97.85%** | 99.2 | 100 | 98.4 | 93.8 |
| 3 | MoT-FM 2-step | MoT | ≈30 | FM | **96.90%** | 98.2 | 97.4 | 98.0 | 94.0 |
| 4 | **DoT-FM 2-step** | DoT | **4** | FM | **94.15%** | 97.6 | 96.2 | 93.6 | 89.2 |
| 5 | DoT-FM continue +2ep freeze | DoT | 4 | FM | **94.20%** | 97.6 | 96.6 | 94.4 | 88.2 |
| 6 | DoT-SF Mixed (foundation) | DoT | 4 | mip mixed | **4.55%** | 0.0 | 0.0 | 14.8 | 3.4 |
| 7 | DoT-SF Mixed from FM freeze | DoT | 4 | mip mixed | **3.20%** | 0.0 | 0.0 | 12.2 | 0.6 |
| — | Alpha foundation (no FT) | MoT | ≈30 | FM | 0% | 0 | 0 | 0 | 0 |

### Fair pair (same DoT-FM init, +2 ep, same freeze/detach/`action_lr=2e-4`)

| Arm | Objective | Overall | Δ vs 94.15% |
|-----|-----------|--------:|------------:|
| FM continue | flow | **94.20%** | **+0.05pp** |
| SF Mixed | mip mixed | **3.20%** | **−90.95pp** |

**Verdict:** freeze/detach is fine. **Hard-switching flow→MIP Mixed destroys the policy.**

---

## 2. Architecture

```
MoT: Video DiT ⟷ Action tokens via joint self-attn, Action depth = video (~30)
DoT: Video DiT → K bridges → shallow ActionDiT cross-attn (K=4, ~214M Action)
```

MoT→DoT warm-start (`finetune_strict_load=false`): **video loads; Action mostly random.**

---

## 3. Hypotheses (not established causes)

1. FM predicts velocity and MIP predicts clean actions; a direct objective switch is a plausible source of instability, **not an established explanation**.
2. The completed DoT-4 MIP-GT-from-foundation run also fails (3.30%), and FM→MIP-GT fails (2.90%). Thus neither Mixed self-conditioning nor an FM-trained action head is necessary for collapse.
3. DoT-8 FM reaches 93.90% versus DoT-4 FM 94.15%. One seed does not establish equivalence or a depth effect.
4. MoT Mixed reaches 97.85% versus MoT FM 96.90%, but its LIBERO-10 result is lower (93.8% versus 94.0%). This does not demonstrate a precision-task improvement.
5. No causal claim about architecture/objective incompatibility is warranted until implementation, video noise conditioning, initialization, and training-seed controls are checked.
6. The continuation pair matches examples seen (546,944), effective batch (128), and recorded optimizer updates (4,274), but differs in ranks/accumulation (FM 8×8×2; MIP 4×8×4) and compute. It is not a measured FLOPs-matched pair.

See also: `docs/autoresearch_shallow_action.md` (paper survey + weekend plan).

---

## 4. Completed weekend results and next evidence

| Run | Successes / trials | Overall | spatial | object | goal | long |
|---|---:|---:|---:|---:|---:|---:|
| DoT-8 FM | 1878 / 2000 | 93.90% | 98.0 | 94.6 | 92.2 | 90.8 |
| DoT-4 MIP-GT foundation | 66 / 2000 | 3.30% | 0.0 | 0.0 | 12.2 | 1.0 |
| DoT-4 FM→MIP-GT | 58 / 2000 | 2.90% | 0.0 | 0.0 | 11.4 | 0.2 |

The per-suite figures above are regenerated/checked against the archived raw results.
See `docs/arxiv_readiness_20261018.md` for the updated research assessment and
`results/research_audit_20261018/runs.csv` for the authoritative counts.

## 5. Artifacts

| Artifact | Path |
|----------|------|
| DoT-FM ckpt | `outputs/experiments/libero_dot_sf_2026-10-13_17-44-16/FM_flow/2026-10-14_14-39-10/` |
| DoT-SF foundation | `.../SF_mip_mixed_resume_from_6000/2026-10-14_10-43-42/` |
| Fair SF from FM | `outputs/experiments/libero_dot_sf_beat_fm_2026-10-15_14-36-26/...` |
| Fair FM continue | `outputs/experiments/libero_dot_fair_fm_vs_sf_2026-10-15_22-40-57/FM_continue_from_FM/` |
| FM→MIP-gt | `outputs/experiments/libero_dot_fm_to_mip_gt_2026-10-16_16-48-07/MIP_gt_from_FM/` |
