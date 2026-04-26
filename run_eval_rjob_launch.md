# rjob: SFT 轨迹采集（`run_eval_rjob.sh`）

项目与 `CONDA` 在 **gpfs2** 的 `wangzifugpfs2` 目录下，提交前请确认 rjob 的 `gpfs://` 卷名与你们平台一致，否则在控制台查 `rjob` 的 mount 文档或让管理员提供 gpfs2 的挂载 URI。

```bash
# 若平台支持多个 --mount，同时挂用户家目录（可选）与 gpfs2 工作区：
rjob submit --name=sft-collect-train-h4-10-noloopTrue --gpu=0 --memory=32000 --cpu=16 --charged-group=ai4cmp_gpu --private-machine=group \
  --mount=gpfs://gpfs1/wangzifu:/mnt/shared-storage-user/wangzifu \
  --mount=gpfs://gpfs2/wangzifugpfs2:/mnt/shared-storage-gpfs2/wangzifugpfs2 \
  --image=registry.h.pjlab.org.cn/ailab/ml-base:22.04-pjlab \
  -e DATASET=train_h4_10 -e MAX_TURNS=100 -e FORCE_NOLOOP=True -e API_MODEL_NAME=A1-preview -e VAL_BATCH_SIZE=32 -e VAL_RESUME=True \
  -e ROLLOUT_N=8 -e ROLLOUT_TEMP=0.7 -e ROLLOUT_TOPP=1.0 -e SFT_SAVE_EVERY=50 -- \
  bash -exc "/mnt/shared-storage-gpfs2/wangzifugpfs2/Agent-r1-temp/run_eval_rjob.sh"

rjob submit --name=sft-collect-train-h4-10-noloopFalse --gpu=0 --memory=32000 --cpu=16 --charged-group=ai4cmp_gpu --private-machine=group \
  --mount=gpfs://gpfs1/wangzifu:/mnt/shared-storage-user/wangzifu \
  --mount=gpfs://gpfs2/wangzifugpfs2:/mnt/shared-storage-gpfs2/wangzifugpfs2 \
  --image=registry.h.pjlab.org.cn/ailab/ml-base:22.04-pjlab \
  -e DATASET=train_h4_10 -e MAX_TURNS=100 -e FORCE_NOLOOP=False -e API_MODEL_NAME=A1-preview -e VAL_BATCH_SIZE=32 -e VAL_RESUME=True \
  -e ROLLOUT_N=8 -e ROLLOUT_TEMP=0.7 -e ROLLOUT_TOPP=1.0 -e SFT_SAVE_EVERY=50 -- \
  bash -exc "/mnt/shared-storage-gpfs2/wangzifugpfs2/Agent-r1-temp/run_eval_rjob.sh"
```

**说明：** 原先用 `/mnt/shared-storage-user/wangzifu/Agent-r1-temp/...` 在 worker 上常不存在。脚本与 `miniconda3` 在 `/mnt/shared-storage-gpfs2/wangzifugpfs2/` 下，因此启动命令与挂载需指向该路径。若你方 **仅** 需要本仓库和 conda，可只保留 `gpfs2` 的 `--mount`（以你们平台可挂载列表为准）。
