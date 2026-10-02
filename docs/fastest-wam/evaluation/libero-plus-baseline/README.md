# LIBERO-Plus public training baseline intake

Acquired 2026-09-26 for Fastest-WAM. The benchmark's official name is **LIBERO-Plus** (the request called it “Libro-Plus”). This directory preserves public source code, exact upstream revisions, Hub metadata with file hashes, and resumable acquisition scripts. The large files are staged on `h100-box` under the existing WAM **training** repository's ignored `data/` tree:

`/root/evan/Fastest-WAM-evan/data/libero-plus-baseline-20260926/`

Nothing here is a WAM training run. OpenVLA-OFT is the authors' released policy baseline; the Fastest-WAM repo already has a separate `benchmarks/libero-plus/` evaluation adapter.

## Sources and pinned versions

| Role | Official source | Revision | Size of main payload |
| --- | --- | --- | ---: |
| Benchmark code and 10,030-task map | [sylvestf/LIBERO-plus](https://github.com/sylvestf/LIBERO-plus) | `4976dc30028e805ff8094b55501d532c48fec182` | 152 MB checkout |
| OpenVLA-OFT training/evaluation code | [moojink/openvla-oft](https://github.com/moojink/openvla-oft) | `e4287e94541f459edc4feabc4e181f537cd569a8` | 1.7 MB checkout |
| Perturbation simulator assets | [Sylvest/LIBERO-plus](https://huggingface.co/datasets/Sylvest/LIBERO-plus) | `dd2bd61b7d9a6fef1abc52d606e983b41886a149` | `assets.zip` 6,395,849,578 bytes |
| Mixed RLDS training trajectories | [Sylvest/libero_plus_rlds](https://huggingface.co/datasets/Sylvest/libero_plus_rlds) | `fb0c7029b076030d5d57227229e4f7460def1f7c` | split ZIP: 75,544,257,300 bytes |
| Starting four-suite OpenVLA-OFT checkpoint | [moojink/openvla-7b-oft-finetuned-libero-spatial-object-goal-10](https://huggingface.co/moojink/openvla-7b-oft-finetuned-libero-spatial-object-goal-10) | `638918f3d1c2e43a39a8a20772bdb8b91835e4b7` | about 16 GB |
| Authors' post-training reference checkpoint | [Sylvest/openvla-7b-oft-finetuned-libero-plus-mixdata](https://huggingface.co/Sylvest/openvla-7b-oft-finetuned-libero-plus-mixdata) | `a85655ec941bae6644c9fbdf62db02b9726d7cf5` | about 16 GB |

The `metadata/*_blobs.json` files are the Hub's pinned public file lists, including exact byte counts and SHA-256 hashes for all large files. `fetch_artifacts.py` checks those hashes; `fetch_parallel.py` handles large files by resumable ranges and checks the same hashes before renaming a `.part` file to its final name. A `.sha256.ok` beside a large file means it passed the independent check.

## What the authors trained

The [paper, §6.2 and Appendix D](https://arxiv.org/html/2510.13626) describes collecting 22,400 candidate trajectories and retaining over 20,000 successful trajectories after filtering, then mixed fine-tuning from official OpenVLA-OFT weights. Its published settings are **8×A100, 2 samples/GPU (global batch 16), 100,000 steps, learning rate 5×10⁻⁴, AdamW with weight decay 0.1, and cosine decay with warmup**. The training set varies objects, backgrounds, lighting, cameras, language and image noise; the paper says the training variations differ from its evaluation scenarios. The released result model's `config.json` names the four-suite checkpoint above as `_name_or_path`.

The authors report **69.6%** overall for the official OpenVLA-OFT baseline and **79.6%** for their augmented post-training checkpoint on LIBERO-Plus. These are published scores, not measurements from this intake.

### Exact reproduction gap

The public `openvla-oft/vla-scripts/finetune.py` is a usable RLDS fine-tuning base, but its current code uses `MultiStepLR` (10× decay at a milestone), leaves AdamW weight decay unspecified, and defaults to zero warmup. The [official OFT LIBERO recipe](https://github.com/moojink/openvla-oft/blob/main/LIBERO.md) describes 150,000 steps and batch 8/GPU for its *own* LIBERO run. The LIBERO-Plus paper does not publish its warmup length or a run-specific training command/patch. The released post-training model also labels its action head and proprio projector `150000_checkpoint`, while Appendix D says 100,000 steps. Keep these as unresolved provenance differences; do not call a run of the untouched public trainer an exact reproduction of the 79.6% result.

## Benchmark and evaluation protocol

The checked-in task classification and suite map agree on **10,030 unique tasks**: spatial 2,402; object 2,518; goal 2,591; long (`libero_10`) 2,519. They cover seven perturbation categories. The [benchmark README](https://github.com/sylvestf/LIBERO-plus#-evaluation) says to replace vanilla LIBERO with this fork and change `num_trials_per_task` from 50 to **1**. Thus the full four-suite evaluation is 10,030 rollouts, scored as success rate overall and by category. The map provides task IDs, categories and difficulty levels; 121 entries have no difficulty level, so preserve that as missing data rather than inventing a level.

The existing Fastest-WAM `benchmarks/libero-plus/` adapter additionally pins seed 10000, global RNG mode, settling steps, and rollout limits for its OpenWAM evaluation. Those are that adapter's explicit choices. To compare with the authors' leaderboard, also match the policy checkpoint, observation cameras/crop, simulator assets, action interface, and per-task rollout behavior. Standard LIBERO's 50-trial results are a different protocol.

## Preparing to train or evaluate

1. Keep OpenVLA-OFT and LIBERO-Plus in isolated Python environments. The [OFT setup](https://github.com/moojink/openvla-oft/blob/main/SETUP.md) calls for Python 3.10, PyTorch, its pinned Transformers fork and FlashAttention 2. The [benchmark README](https://github.com/sylvestf/LIBERO-plus#-installation) has separate simulator dependencies and requires extracting `assets.zip` into `libero/libero/assets/`. Its `import libero` replaces vanilla LIBERO; mixing both installs in one environment can select the wrong simulator.
2. Download and SHA-verify all artifacts under the H100 `data/libero-plus-baseline-20260926/artifacts/` directory. The small configs and model cards can be acquired with `fetch_artifacts.py --group all --max-bytes 10000000`; use `fetch_parallel.py` for each large LFS file. Both scripts read the pinned metadata in this directory. The H100's existing `/root/openwam-eval/env/bin/python` supplies `huggingface_hub` and `requests` for **downloading only**; no evaluation service or package in that environment was changed.
3. Extract the split RLDS ZIP and inspect its top-level dataset name and feature schema before setting `--data_root_dir` and `--dataset_name` in `vla-scripts/finetune.py`. The archives are a distribution format, not a directly trainable directory. The precise extraction and trainer compatibility check remain to be done after the verified archive completes.
4. Run an isolated short dataloader smoke check before allocating GPUs for training. For the authors' exact recipe, resolve the warmup schedule and the 100k/150k checkpoint naming discrepancy first. Use the authors' released checkpoint as an evaluation reference meanwhile.

## Intake checks

- Both Git checkouts match the SHAs above locally and on H100.
- Static check of `task_classification.json` against `libero_suite_task_map.py`: all four suite counts and names match exactly; 10,030 names total; no duplicates or missing mapped names.
- Download helper syntax compiles; a 499,723-byte tokenizer file was fetched by range and independently SHA-256 verified on H100.
- A GPU training run and simulator rollout have **not** been started by this intake. The Mac has no NVIDIA environment; H100 setup must remain isolated from its active WAM training and evaluation services.

See `DOWNLOAD_STATUS.md` for the latest staged-file status and exact remaining work.
