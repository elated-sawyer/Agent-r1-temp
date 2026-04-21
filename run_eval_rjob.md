rjob 提交命令

rjob submit \
  --name=eval-ppo-retro \
  --gpu=2 \
  --memory=64000 \
  --cpu=64 \
  -P 2 \
  --charged-group=agentsft_gpu \
  --private-machine=group \
  --mount=gpfs://gpfs1/wangzifu:/mnt/shared-storage-user/wangzifu \
  --mount=gpfs://gpfs2/gpfs2-shared-public:/mnt/shared-storage-gpfs2/gpfs2-shared-public \
  --mount=gpfs://gpfs1/ai4cmp:/mnt/shared-storage-user/ai4cmp \
  --image=registry.h.pjlab.org.cn/ailab/ml-base:22.04-pjlab \
  -e DISTRIBUTED_JOB=true \
  -e DATASET=retro \
  -e MODEL_PATH=/mnt/shared-storage-gpfs2/gpfs2-shared-public/huggingface/hub/models--Qwen--Qwen3.5-35B-A3B-Base/snapshots/e2ee613a3820d4482636d51b6d2eec45eb6d025b \
  -e API_MODEL_NAME='Qwen/Qwen3.5-35B-A3B' \
  -e MAX_TURNS=100 \
  -e TOPK=10 \
  -e MAXSTEP=30 \
  -e API_MAX_CONCURRENCY=50 \
  -e HF_HUB_OFFLINE=1 \
  -e pjlab_APImodel_key="$pjlab_APImodel_key" \
  -e pjlab_APImodel_url="$pjlab_APImodel_url" \
  -e CONDA_SH=/mnt/shared-storage-user/wangzifu/miniconda3/etc/profile.d/conda.sh \
  -e CONDA_ENV=Retro_R1 \
  -- bash -exc "/mnt/shared-storage-user/wangzifu/Agent-r1-temp/run_eval_rjob.sh"
几点注意：

CONDA_SH 里请填你在 gpfs1 上真实的 conda 路径；若镜像本身就自带 Python 依赖，删掉 CONDA_SH/CONDA_ENV 两行即可走系统 python3。
pjlab_APImodel_key/url 需要在本地 shell 已经 export，否则把值直接写在命令里。
如需跑 chembl 数据集：改成 -e DATASET=chembl。