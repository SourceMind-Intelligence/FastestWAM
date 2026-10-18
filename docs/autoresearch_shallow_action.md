> Superseded by the [2026-10-18 evidence audit](arxiv_readiness_20261018.md). The causal claims and predicted benefits below were working hypotheses; completed weekend results do not support several of them.

# AutoResearch: Better Action Generation on Shallow DoT

**Date:** 2026-10-16 · **GPU window:** through this weekend · **Branch:** `tengfei`

## 1. Empirical facts (LIBERO 40×50, 2-step NFE)

| Run | Arch | Objective | Overall | Note |
|-----|------|-----------|--------:|------|
| Alpha sim-libero | MoT ~30L | FM | **99.55%** | upper bound on this data |
| MoT-SF Mixed | MoT | MIP mixed | **97.85%** | SF *can* beat FM when Action is deep + mature |
| MoT-FM | MoT | FM | **96.90%** | |
| **DoT-FM** | DoT 4L | FM | **94.15%** | strong shallow baseline |
| DoT-FM continue +2ep (freeze) | DoT 4L | FM | **94.20%** | freeze recipe OK |
| DoT-SF Mixed from foundation | DoT 4L | MIP mixed | **4.55%** | random Action + Mixed → dead |
| DoT-SF Mixed from FM (freeze) | DoT 4L | MIP mixed | **3.20%** | objective switch destroys policy |
| DoT FM→MIP-gt soft | DoT 4L | MIP **gt** | **eval running** | soft convert, no freeze, low LR |

**Root cause of FM→Mixed collapse:** FM Action head predicts **velocity**; MIP trains **clean A (x0)**. Incompatible output semantics; Mixed then feeds garbage `pred0` into refine.

## 2. Related work (what to steal)

| Paper | Idea | Relevance to us |
|-------|------|-----------------|
| [Consistency Policy](https://arxiv.org/abs/2405.07503) | Distill multi-step DP teacher → few-step student (CTM) | **Best match:** keep DoT-FM as teacher; distill 1–2 step Action student *without* changing to MIP param |
| [OneDP](https://proceedings.mlr.press/v267/wang25ba.html) | KL distill along diffusion chain → 1-step generator | Same teacher–student path; 2–10% extra train cost |
| [MP1 MeanFlow](https://arxiv.org/abs/2507.10543) / [DM1](https://arxiv.org/html/2510.07865v1) | Learn **interval-averaged** velocity → true 1-NFE | Stay in velocity space (compatible with FM weights); avoid MIP x0 switch |
| [FlowPolicy](https://arxiv.org/html/2412.04987) | Consistency-FM without distillation | Train-from-scratch few-step flow on Action |
| [Self-GAD](https://arxiv.org/abs/2508.12189) / [SGAC](https://arxiv.org/abs/2510.12392) | Inference-time self-guidance / adaptive chunking | **Zero train cost** boost on top of DoT-FM |
| Hybrid Consistency Policy | SDE prefix + one consistency jump | Hybrid if pure 1-step loses multimodality |

### Design principle for FastestWAM DoT

1. **Do not hard-switch FM→MIP Mixed on a velocity-trained head.**
2. Prefer methods that **stay in velocity / flow space** (MeanFlow, Consistency-FM) or **distill a student** that is trained from scratch for 1-NFE with FM teacher targets.
3. Use **inference-time guidance** as a free win while training heavier methods.
4. If capacity is the bottleneck vs MoT (94% vs 97%), try **deeper DoT (8–12 layers)** with FM first — cheap ablation.

## 3. Weekend experiment queue (keep GPUs busy)

| Priority | Job | Est. wall (8×H100) | Success criterion |
|----------|-----|--------------------|-------------------|
| P0 | Eval FM→MIP-gt | ~6–10h | ≥90% → MIP soft convert works; else abandon MIP on DoT |
| P1 | DoT-8 FM from foundation, 4ep | ~8–12h | >94.15% → capacity helps |
| P2 | DoT-4 MIP-**gt** from foundation, 4ep | ~10–14h | Fair FM vs MIP (same init, no Mixed) |
| P3 | Inference self-guidance on DoT-FM (code) | code + short eval | +pp without train |
| P4 | Consistency / MeanFlow Action (code+train) | 1–2 days | Beat FM-2 at 1–2 NFE in velocity space |

Auto pipeline: `scripts/experiments/run_weekend_autoresearch_pipeline.sh`

## 4. Artifact index

| Item | Path |
|------|------|
| DoT-FM 94.15% | `outputs/libero/dot_FM_flow_final_2step/summary.json` |
| Fair FM continue 94.20% | `outputs/libero/dot_FM_continue_from_FM_2step/summary.json` |
| Fair SF from-FM 3.20% | `outputs/libero/dot_SF_from_FM_freeze_2step/summary.json` |
| FM→MIP-gt ckpt | `outputs/experiments/libero_dot_fm_to_mip_gt_*/MIP_gt_from_FM/.../checkpoint_step_8546.safetensors` |
| Results analysis | `docs/libero_dot_sf_results_analysis.md` |
