# Shallow ActionDiT: beating Flow Matching

## Evidence so far (LIBERO, 2-step NFE-matched)

| Run | Action init | Overall |
|-----|-------------|---------|
| DoT-FM (`joint_cross_attn`, 4-layer Action) | trained with flow | **94.15%** |
| DoT-SF Mixed MIP-2 from MoT foundation | Action mostly random | **~4%** (34/40 tasks) |

SF does not bootstrap a random 4-layer ActionDiT. Mixed MIP needs a usable `pred_0`.

## Active experiment (saturate idle GPUs 2,3,6,7)

Script: `scripts/experiments/run_libero_dot_sf_beat_fm.sh`

1. Warm-start from DoT-FM `checkpoint_step_17092`
2. Switch objective → Mixed SF MIP-2 (`t∈{0,0.9}`)
3. Freeze `video_backbone.dit` + `detach_bridge=true` (Action-only updates; LoRA-equivalent focus without new infra)
4. `action_lr=2e-4`, global batch 128 on 4 GPUs, 2 epochs

Success criterion: LIBERO 2-step overall **> 94.15%** (beat FM).

## Follow-ups if this plateaus ≤ FM

- Soft video adapt: unfreeze DiT with `video_lr=1e-5` or true LoRA on DiT linears
- `mip_refine_mode=gt` only (stabler teacher) from the same FM ckpt
- Consistency / 1-step distill from the FM teacher onto the shallow ActionDiT
