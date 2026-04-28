rjob submit --name=test-eval-chembl-true --gpu=0 --memory=8000 --cpu=8 --charged-group=ai4cmp_gpu --private-machine=group --mount=gpfs://gpfs2/wangzifugpfs2:/mnt/shared-storage-gpfs2/wangzifugpfs2 --image=registry.h.pjlab.org.cn/ailab/ml-base:22.04-pjlab -e DATASET=chembl -e MAX_TURNS=100 -e FORCE_NOLOOP=True -e API_MODEL_NAME=A1-preview -e VAL_BATCH_SIZE=128 -e VAL_RESUME=True -- bash -exc "/mnt/shared-storage-user/wangzifu/Agent-r1-temp/run_eval_rjob.sh"

rjob submit --name=test-eval-chembl-false --gpu=0 --memory=8000 --cpu=8 --charged-group=agentsft_gpu --private-machine=group --mount=gpfs://gpfs2/wangzifugpfs2:/mnt/shared-storage-gpfs2/wangzifugpfs2 --image=registry.h.pjlab.org.cn/ailab/ml-base:22.04-pjlab -e DATASET=chembl -e MAX_TURNS=100 -e FORCE_NOLOOP=False -e API_MODEL_NAME=A1-preview -e VAL_BATCH_SIZE=128 -e VAL_RESUME=False -- bash -exc "/mnt/shared-storage-user/wangzifu/Agent-r1-temp/run_eval_rjob.sh"



MOUNT_WANGZIFU="${MOUNT_WANGZIFU:-gpfs://gpfs1/wangzifu:/mnt/shared-storage-user/wangzifu}"
MOUNT_GPFS2_PUBLIC="${MOUNT_GPFS2_PUBLIC:-gpfs://gpfs2/gpfs2-shared-public:/mnt/shared-storage-gpfs2/gpfs2-shared-public}"
MOUNT_AI4CMP="${MOUNT_AI4CMP:-gpfs://gpfs1/ai4cmp:/mnt/shared-storage-user/ai4cmp}"
MOUNT_GPFS2_WZF="${MOUNT_GPFS2_WZF:-gpfs://gpfs2/wangzifugpfs2:/mnt/shared-storage-gpfs2/wangzifugpfs2}"
