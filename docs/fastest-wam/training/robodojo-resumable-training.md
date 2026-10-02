# h100-box RoboDojo 可恢复训练（2026-09-27 启动）

## 上次 pilot 的教训

2026-09-17 的 RoboDojo pilot 在约 step 24465 停止，最后完整权重为 step 24000。那次运行设定 `save_full_states_for_resume=false`，因此只剩模型权重，没有优化器、学习率调度器和随机状态。旧运行不能做严格续训；使用其 step-24000 权重只能热启动一个新实验，步数与优化器状态重新开始。证据保存在 H100 的 `/root/evan/Fastest-WAM-evan/outputs/openwam_checkpoints/2026-09-17_15-32-18/RESUME_NOTES.md`。

## 本次任务

- 训练仓库：`/root/evan/Fastest-WAM-evan/`，分支 `MIP`，2026-09-28 启动前 HEAD `5210d55`。
- 模型：从上述 step-24000 权重热启动，显式选用 `action_objective=flow`。这不是旧运行的严格续训。
- 数据：现有 `dataloader=robodojo`，`arx_x5` 仿真数据；训练 5 epoch、每 GPU batch 16、随机种子 42。
- 本次启动时使用 GPU 0–5、7。按用户 2026-09-28 的新约定，后续任务默认可使用 GPU 6；历史累计 ECC 计数不构成排除依据。本次任务的 GPU 拓扑不在运行中变更。
- 在 step 10 先保存一次模型权重及完整 Accelerate/DeepSpeed 状态，此后每 1000 步保存；保留最近两个保存点。正常结束时也保留最后的完整状态。
- 恢复预检已完成：step 2 的权重和约 103 GB 完整状态写入成功，从 step 2 实际恢复并完成 step 3，step 3 完整状态也已保存。随后直接启动正式 5 epoch 训练。
- 截至 2026-09-28 23:20 CST，正式任务 PID `443359` 仍在运行，进度为 `37478/82865`（约 45%，第 3 个 epoch；日志记为 `epoch=2`）。最近完整恢复点为 step 37000；step 36000 和 37000 的权重及训练状态均在。本次使用的 7 张 H100 显存占用约 61 GB/卡，查询时 GPU 利用率为 99–100%。
- 2026-09-28 23:22 CST 应用户要求暂停：先验证 step 37000 的完整状态标记与对应权重，再终止训练进程。停止前日志到 step 37537；恢复时从 step 37000 继续，最多重做 537 个尚未保存的步骤。终止后 8 张 GPU 显存占用均为 0，未保留自动重启队列。

## 远端位置与恢复

- 启动参数：`scripts/experiments/robodojo_resumable_20260928.sh`
- 预检与原计划安全接续脚本：`scripts/experiments/run_robodojo_after_eval_20260928.sh`（评测释放后改为直接启动；排队器已停止）
- 状态：`logs/robodojo-resumable-20260928/status.txt`
- 排队与训练日志：同目录下的 `queue.log`、`smoke-start.log`、`smoke-resume.log`、`train.log`
- 正式训练目录指针：`logs/robodojo-resumable-20260928/train-run-dir.txt`
- 本次正式训练目录：`outputs/openwam_checkpoints/robodojo_5epoch_resumable_20260928/2026-09-28_00-51-21/`；其中 `run_provenance/` 保存本次代码差异、启动脚本、Git 版本和校验值。

若正式任务中断，先确认上述目录下有 `accel_state_step_<N>/trainer_state.json` 和对应权重，再在该训练仓库运行：

```bash
run_dir=$(cat logs/robodojo-resumable-20260928/train-run-dir.txt)
scripts/experiments/robodojo_resumable_20260928.sh train-resume "$run_dir"
```

`trainer_state.json` 是完整状态写完后的原子完成标记；半写入目录不能视为有效恢复点。恢复时复用原训练目录，不能把 `finetune_ckpt_path` 当成严格恢复。若改变数据、批大小、GPU 数或训练设置，先单独评估恢复兼容性。
