# Zero-input temporal symmetry in the local cross-attention action head

This note describes a property of **this implementation**, not a universal limitation of MIP or shallow policies. The closed-loop rescue experiment is ongoing. See [the research audit](arxiv_readiness_20261018.md) and [the new evidence directory](../results/last4h_20261018/).

## Proposition

Consider a sequence model with these properties:

1. The initial action embedding is a shared tokenwise map, with no absolute position or learned query embedding.
2. Action self-attention applies position only to Q/K; V is a shared function of the token representation.
3. Cross-attention has position-free action queries and a shared observation K/V bank.
4. Normalization, feed-forward layers, residuals, and timestep modulation are shared over action positions. Timestep is constant across the chunk. There is no position-specific stochastic operation.

If every input action token is identical, every output action token is identical in exact arithmetic, at every depth.

**Proof.** Suppose all input tokens at a block equal h. Then all value vectors in action self-attention equal v. RoPE can change the attention weights at each query, but normalized attention still produces

\[
\sum_j \alpha_{ij}v=v\quad\text{because}\quad\sum_j\alpha_{ij}=1.
\]

Thus self-attention preserves equality. Cross-attention receives identical queries, hence identical attention weights over the common observation memory and identical outputs. Shared tokenwise operations preserve equality as well. Residual additions preserve equality. Apply this argument inductively through the block stack and the shared decoder. Multihead attention has the same property head by head. Floating-point roundoff does not supply a reliable positional signal. ∎

For deterministic MIP, the initial input is zero at every action position. If the first prediction is a constant sequence, multiplying it by t* and refining with no added noise preserves the premise. Therefore the entire inference chain lies in the space of temporally constant action chunks. Increasing the action depth alone does not remove this constraint.

This does **not** mean the model is independent of its observation: it can output different constant chunks for different observations. Executing only the first action and replanning can still yield useful control. The restriction matters especially for executing a multi-action chunk, and for supervised future-trajectory fitting.

## Why the training loss can mislead

GT refinement supplies a time-varying target-corrupted sequence during training. Mixed refinement also inherits time variation from its GT component. Both can give the refiner information that the deterministic inference chain cannot produce. The observed small second-branch loss therefore does not certify useful inference-time refinement.

A temporal-constant predictor has an irreducible per-window masked squared error. Let a_td be the normalized action target and w_td its binary validity mask. For a constant prediction c_d, the oracle optimum is

\[
 c_d^*=\frac{\sum_t w_{td}a_{td}}{\sum_t w_{td}},\qquad
 L_{\rm const}^*=\frac{\sum_{td}w_{td}(a_{td}-c_d^*)^2}{\sum_{td}w_{td}}.
\]

Dimensions with no valid targets contribute zero and use a clamped denominator in code. An observation-conditioned policy does not see the future actions, so this oracle is an optimistic lower bound, not an achievable policy baseline.

## Evidence available now

| Check | Result | Scope |
|---|---|---|
| Random 1/4/8-layer heads, random observation memory | Zero-input coarse/refine token differences about 1e-7 in fp32 | Unit-level architectural evidence |
| Trained DoT-4 MIP-GT head, synthetic memory, fp32 | Max coarse 2.61e-7; refine 2.83e-7 | Actual trained action weights, not real visual features |
| Same trained head, CPU bf16 | Coarse and refine temporal differences exactly zero | Same synthetic test |
| Same weights, added position scale0.1 | fp32 coarse/refine differences 0.140 / 0.215 | Breaks constraint; no success-rate claim |
| 1,024 sampled LIBERO training windows | Oracle constant MSE 0.05487 | Normalized, masked targets; samples can overlap |
| Historical trained coarse branch | Last-100-microbatch mean about 0.053–0.055 | Different sampled windows from the oracle diagnostic |

Synthetic conditioned sine/cosine trajectory regression isolates the representational issue. Both arms use identical initial weights, training batches, and optimizer settings; only the positional scale differs. There are 128 training and 64 test observations, 32 action positions, two action dimensions, two layers, and 500 AdamW updates.

| Seed | No position: test MSE | Sinusoidal scale0.1: test MSE | Constant oracle MSE |
|---:|---:|---:|---:|
| 0 | 0.14456 | 0.00353 | 0.14326 |
| 1 | 0.15113 | 0.00433 | 0.14944 |
| 2 | 0.15128 | 0.00400 | 0.15009 |

These synthetic results demonstrate learnability after breaking the constraint. They are not additional robot tasks or independent LIBERO training seeds.

## Repair and causal experiment

`action_position_scale=0` preserves old behavior and state-dict compatibility. A positive value adds standard fixed sinusoidal positions to the cross-attention action embeddings. It introduces no new trainable parameters. The current short-run choice, 0.1, is a diagnostic setting rather than a tuned optimum.

The two real-model arms start from the same trained DoT-4 MIP-GT checkpoint, freeze VideoDiT, detach bridges, and train the action branch for 1,500 microbatches with seed42, 2 GPUs, batch4, accumulation2. This is 12,000 sample presentations / 750 nominal optimizer updates, not a full fine-tuning run. Evaluation is 40 tasks × 5 trials. Runtime logs are authoritative about actual updates and whether a run completed.

A successful rescue would strengthen the explanation; it would not establish that all differences between MoT and cross-attention are caused by positional symmetry. Remaining factors include video noise conditioning, warm-start weight transfer, context fusion, longer training, action horizon, and evaluation randomness. Failure of a short adaptation run would not refute the algebraic constraint either.

Alternatives worth comparing after the minimal repair are learned action queries, position-aware bridge attention, and explicitly stochastic starts with matched train/inference semantics. Each changes a different part of the model and needs its own control.

## Reproduction

```bash
python -m pytest -q tests/test_mip_temporal_symmetry.py
python scripts/experiments/diagnose_mip_temporal_symmetry.py \
  --checkpoint <trained_DoT_MIP_checkpoint.safetensors> \
  --output /tmp/symmetry.json
python scripts/experiments/measure_constant_action_lower_bound.py \
  --output /tmp/constant_bound.json
python scripts/experiments/toy_mip_position_regression.py \
  --output /tmp/toy_position.json
```

The checkpoint diagnostic defaults to the Wan2.2-5B geometry used here; other backbones need the matching head and layer settings.

## Relation to prior work

[MIP](https://arxiv.org/abs/2512.01809) motivates the deterministic two-step regression chain. [Faster-WAM](https://arxiv.org/abs/2608.02365) already includes explicit positional alignment in its world/action interaction design. Our local bridge implementation is not its full architecture. The symmetry argument is a direct consequence of normalized attention, and we make no claim that adding positional embeddings is a novel technique. The potential contribution is a carefully validated failure analysis of objective transfer in a pretrained world–action stack.
