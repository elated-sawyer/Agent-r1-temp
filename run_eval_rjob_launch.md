# rjob: SFT 轨迹采集（`run_eval_rjob.sh`）

项目与 `CONDA` 在 **gpfs2** 的 `wangzifugpfs2` 目录下，提交前请确认 rjob 的 `gpfs://` 卷名与你们平台一致，否则在控制台查 `rjob` 的 mount 文档或让管理员提供 gpfs2 的挂载 URI。

```bash
# 若平台支持多个 --mount，同时挂用户家目录（可选）与 gpfs2 工作区：
rjob submit --name=sft-collect-train-h4-10-noloopTrue-noback --gpu=0 --memory=32000 --cpu=16 --charged-group=ai4cmp_gpu --private-machine=group \
  --mount=gpfs://gpfs1/wangzifu:/mnt/shared-storage-user/wangzifu \
  --mount=gpfs://gpfs2/wangzifugpfs2:/mnt/shared-storage-gpfs2/wangzifugpfs2 \
  --image=registry.h.pjlab.org.cn/ailab/ml-base:22.04-pjlab \
  -e DATASET=train_h4_10 -e BACKTRACK=false -e MAX_TURNS=100 -e FORCE_NOLOOP=True -e API_MODEL_NAME=A1-preview -e VAL_BATCH_SIZE=32 -e VAL_RESUME=True \
  -e ROLLOUT_N=8 -e ROLLOUT_TEMP=0.7 -e ROLLOUT_TOPP=1.0 -e SFT_SAVE_EVERY=50 -- \
  bash -exc "/mnt/shared-storage-gpfs2/wangzifugpfs2/Agent-r1-temp/run_eval_rjob.sh"

rjob submit --name=sft-collect-train-h4-10-noloopFalse-noback --gpu=0 --memory=32000 --cpu=16 --charged-group=ai4cmp_gpu --private-machine=group \
  --mount=gpfs://gpfs1/wangzifu:/mnt/shared-storage-user/wangzifu \
  --mount=gpfs://gpfs2/wangzifugpfs2:/mnt/shared-storage-gpfs2/wangzifugpfs2 \
  --image=registry.h.pjlab.org.cn/ailab/ml-base:22.04-pjlab \
  -e DATASET=train_h4_10 -e BACKTRACK=false -e MAX_TURNS=100 -e FORCE_NOLOOP=False -e API_MODEL_NAME=A1-preview -e VAL_BATCH_SIZE=32 -e VAL_RESUME=True \
  -e ROLLOUT_N=8 -e ROLLOUT_TEMP=0.7 -e ROLLOUT_TOPP=1.0 -e SFT_SAVE_EVERY=50 -- \
  bash -exc "/mnt/shared-storage-gpfs2/wangzifugpfs2/Agent-r1-temp/run_eval_rjob.sh"

# 可回溯对照组（BACKTRACK=true 走 main_agent_retro + tool.env=retro，启用 back_state 工具）。
# 注意：BACKTRACK=true 时 FORCE_NOLOOP 仅作为 Hydra 配置被写入但不被读取。
rjob submit --name=sft-collect-train-h4-10-noloopTrue-back --gpu=0 --memory=16000 --cpu=8 --charged-group=ai4cmp_gpu --private-machine=group \
  --mount=gpfs://gpfs1/wangzifu:/mnt/shared-storage-user/wangzifu \
  --mount=gpfs://gpfs2/wangzifugpfs2:/mnt/shared-storage-gpfs2/wangzifugpfs2 \
  --image=registry.h.pjlab.org.cn/ailab/ml-base:22.04-pjlab \
  -e DATASET=train_h4_10 -e BACKTRACK=true -e MAX_TURNS=100 -e API_MODEL_NAME=A1-preview -e VAL_BATCH_SIZE=32 -e VAL_RESUME=True \
  -e ROLLOUT_N=8 -e ROLLOUT_TEMP=0.7 -e ROLLOUT_TOPP=1.0 -e SFT_SAVE_EVERY=50 -- \
  bash -exc "/mnt/shared-storage-gpfs2/wangzifugpfs2/Agent-r1-temp/run_eval_rjob.sh"
```

**`BACKTRACK` 开关说明：**
- `BACKTRACK=false`（默认）：使用 `main_agent_retro_noback` + `tool.env=retro_noback_V4`，**不**暴露 `back_state` 工具；`FORCE_NOLOOP` 生效。
- `BACKTRACK=true`：使用 `main_agent_retro_back` + `tool.env=retro`，**启用** `back_state` 工具，允许 agent 回溯到历史 state。Checkpoint 目录与日志文件名会自动加上 `back` / `noback` 标签，互不覆盖。
- 两个分支都走**精简的验证-only 入口**（`ValidationPipeline`，无 Ray / actor / critic），因此 `run_eval_rjob.sh` 里没有 `actor_rollout_ref.actor.*` / `critic.*` 的配置也能直接跑。
- **不要**把 `BACKTRACK=true` 指向 `agent_r1.src.main_agent_retro`（那是完整 PPO 训练入口，会 Ray + 读 `config.actor_rollout_ref.actor.strategy` / `critic.strategy`，此脚本没有提供这些字段会直接报错）。
- `BACKTRACK=true` 时 `tool.force_noloop` 仍会被写入 Hydra 配置，但 `ToolEnvRetro` 不接受该字段，只有 `ValidationPipeline` 会读它用于保存路径命名，不影响评测语义。

**说明：** 原先用 `/mnt/shared-storage-user/wangzifu/Agent-r1-temp/...` 在 worker 上常不存在。脚本与 `miniconda3` 在 `/mnt/shared-storage-gpfs2/wangzifugpfs2/` 下，因此启动命令与挂载需指向该路径。若你方 **仅** 需要本仓库和 conda，可只保留 `gpfs2` 的 `--mount`（以你们平台可挂载列表为准）。
