# RoboDojo

OpenWAM fully supports **training** on RoboDojo — both the simulation corpus and the
real-robot corpora — through its standard training pipeline:

```bash
python scripts/download_assets/download_benchmark_data.py   # RoboDojo / RoboDojo-Real
bash scripts/train.sh dataloader=robodojo
```

See `configs/dataloader/robodojo.yaml` for the dataloader configuration (sim vs. real
variants and embodiments) and the root README for the general training workflow.

**Evaluation** does not live in this repository. RoboDojo evaluation runs through
[XPolicyLab](https://github.com/XPolicyLab/XPolicyLab) — please head over to that
repository and follow its instructions for environment installation and evaluation.

## Reproducing the OpenWAM-α baseline

Needs a CUDA host with RoboDojo (Isaac Sim 5.1) and XPolicyLab's `policy/OpenWAM`
adapter installed, plus the released `OpenWAM-Alpha-Sim-RoboDojo` checkpoint
(~25 GB) symlinked into the adapter's `checkpoints/`.

```bash
XPL_OPENWAM_DIR=/path/to/RoboDojo/XPolicyLab/policy/OpenWAM \
POLICY_ENV=openwam EVAL_ENV=robodojo POLICY_GPU=0 ENV_GPU=1 \
  bash benchmarks/robodojo/run_openwam_baseline.sh       # 54 task dirs x seeds 0 1 2

python benchmarks/robodojo/compare_to_reference.py /path/to/RoboDojo/eval_result/RoboDojo
```

The comparison aggregates exactly like RoboDojo's `summarize_result.py` and checks
each task against `assets/robodojo_verification/openwam_robodojo_per_seed.json`
with a pooled two-proportion z-test (|z| <= 3 counts as within noise) and the
Avg SR against 11.92 (tolerance 1.5 points). It works on partial runs too.

## Results

Scores from the OpenWAM paper. **Bold** = best, <u>underline</u> = second best; Type distinguishes WAM vs VLA. Column groups: GenStd/GenRand = generalization (standard/randomized), Prec = precision, LongH = long-horizon, Mem = memory, Open = open tasks; SR / Sc = success rate / progress score.

| Method | Type | GenStd SR | GenStd Sc | GenRand SR | GenRand Sc | Prec SR | Prec Sc | LongH SR | LongH Sc | Mem SR | Mem Sc | Open SR | Open Sc | Avg SR | Avg Sc |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| StarVLA-α | VLA | 5.00 | 7.54 | 0.00 | 0.33 | 4.33 | 9.90 | 6.50 | 14.15 | 2.44 | 3.34 | 0.58 | 0.68 | 3.24 | 6.40 |
| X-VLA | VLA | 12.00 | 17.90 | 1.00 | 3.04 | 12.00 | 18.32 | 9.75 | 16.53 | 3.56 | 4.76 | 0.50 | 0.55 | 6.52 | 10.13 |
| π₀.₅ | VLA | 15.00 | 20.93 | 1.00 | 5.82 | 5.50 | 12.40 | 14.67 | 23.54 | 4.56 | 5.78 | 1.67 | 1.98 | 6.91 | 11.41 |
| Spatial Forcing | VLA | 15.00 | 21.25 | 4.00 | 6.98 | 10.58 | 17.33 | 14.58 | 23.26 | 4.11 | 5.43 | 1.58 | 1.78 | 8.04 | 12.38 |
| Hy-Embodied-0.5-VLA | VLA | 17.00 | 21.98 | 0.00 | 1.57 | 8.00 | 13.81 | 14.92 | 25.74 | <u>12.11</u> | <u>13.37</u> | 0.58 | 0.65 | 8.80 | 13.07 |
| Xiaomi-Robotics-1 | VLA | **28.00** | **35.65** | **6.00** | **11.44** | <u>18.83</u> | <u>26.69</u> | 23.67 | <u>38.39</u> | 6.56 | 7.81 | **3.58** | **3.94** | 13.93 | 20.07 |
| Galaxea G0.5 | VLA | 20.00 | 26.74 | **6.00** | <u>11.16</u> | **20.42** | **28.25** | **32.25** | **44.12** | 7.33 | 8.61 | 1.58 | 1.73 | <u>14.88</u> | <u>20.23</u> |
| DM0.5 | VLA | 18.00 | 23.49 | 4.00 | 8.06 | 16.75 | 24.82 | 19.50 | 33.70 | **47.44** | **47.74** | <u>2.08</u> | <u>2.43</u> | **19.34** | **24.90** |
| Fast-WAM | WAM | 2.00 | 4.33 | 0.00 | 0.34 | 0.00 | 1.96 | 5.17 | 9.14 | 3.44 | 3.55 | 0.42 | 0.42 | 2.03 | 3.48 |
| AHA-WAM | WAM | 6.00 | 10.32 | 0.00 | 1.26 | 2.42 | 5.86 | 2.67 | 8.61 | 2.78 | 2.97 | 0.83 | 0.88 | 2.39 | 4.82 |
| GigaWorld-Policy | WAM | 6.00 | 10.28 | 0.00 | 0.41 | 1.83 | 6.15 | 8.92 | 15.51 | 2.22 | 3.46 | 0.50 | 0.54 | 3.27 | 6.20 |
| X-WAM | WAM | 5.00 | 11.24 | 1.00 | 3.54 | 1.83 | 6.72 | 9.08 | 17.47 | 4.67 | 6.32 | 0.25 | 0.57 | 3.83 | 7.69 |
| **OpenWAM-α** | WAM | <u>25.56</u> | <u>33.16</u> | <u>4.11</u> | 8.26 | 9.25 | 18.45 | <u>25.33</u> | 34.93 | 9.11 | 10.41 | 1.08 | 1.41 | 11.92 | 17.18 |
