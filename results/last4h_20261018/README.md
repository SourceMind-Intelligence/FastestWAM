# Live evidence from the final GPU window

This directory is refreshed locally every minute by `collect_last4h_evidence.py`.
Read `status.json` for coverage and completion flags. All new rollouts are diagnostics, not canonical 40×50 evaluations. Partial rates cover completed tasks only. The original MoT/GPU6-render arm is explicitly excluded after rendering failures; its independent retry is named `render2`.

- `flops/`: completed batch1, four-rank six-microbatch calibration, scalar logs and configuration.
- `rescue_training/`: matched continuation with positional scale 0 versus 0.1, ongoing until final step.
- Per evaluation: source manifest, task records, episode CSV and coverage-aware summary.
- `latency/`: populated only if the queued GPU0 timing benchmark executes successfully.

No network push is performed by the watcher. Checkpoints remain under `outputs/`.
