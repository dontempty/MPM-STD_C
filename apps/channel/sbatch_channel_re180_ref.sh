#!/bin/bash
# MPM-STD channel — GPU 노드 (gpu01 40GB / gpu02 80GB 어느 쪽이든) GPU 런.
#   sbatch -J <이름> apps/channel/sbatch_gpu02_channel.sh <input.ini>
# 환경은 Filtered_TDMA/env_gpu_local.sh 로 검증된 조합 그대로:
# hpcx-2.15 의 ucx/ompi 를 시스템 /lib UCX 1.14 보다 앞세운다.
#SBATCH -p batch
# 노드 고정 없음 — gpu01(40GB)·gpu02(80GB) 어느 쪽이든 먼저 비는 곳에서 돈다
# (768x512x384 도 GPU 당 ~11GB 라 40GB 에 들어간다).
#SBATCH -N 1
#SBATCH -n 2
#SBATCH --gres=gpu:2
#SBATCH -o /shared/home/wel1come1234/workspace/MPM-STD/apps/channel/log/channel_re180_ref_gpu_%j.log

set -u
ROOT=/shared/home/wel1come1234/workspace/MPM-STD
NVR=/opt/nvidia/hpc-sdk/Linux_x86_64/23.7
HPCX=$NVR/comm_libs/12.2/hpcx/hpcx-2.15
export OPAL_PREFIX=$HPCX/ompi
export PATH=$NVR/cuda/12.2/bin:$HPCX/ompi/bin:$NVR/compilers/bin:$PATH
export LD_LIBRARY_PATH=$HPCX/ucx/lib:$HPCX/ompi/lib:$NVR/cuda/12.2/lib64:$NVR/math_libs/12.2/lib64:$NVR/compilers/lib:${LD_LIBRARY_PATH:-}
unset PMIX_INSTALL_PREFIX
export OMPI_MCA_opal_warn_on_missing_libcuda=0

cd "$ROOT"
echo "# host=$(hostname) gpus=$(nvidia-smi -L | wc -l)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# ① 스모크 (32x24x32, 200 스텝, ~1분) — GPU 경로 검증. 실패하면 본런 안 간다.
echo "=== SMOKE: input_channel_smoke_budget.ini ==="
"$HPCX/ompi/bin/mpirun" -np 2 ./build_gpu/bin/channel     apps/channel/input_channel_smoke_budget.ini || { echo "SMOKE FAILED"; exit 1; }

# ② Re_tau 180 참조박스 본런 (256^3, 4pi x 2pi x 2, 60000 스텝) — 같은 할당에서 바로 이어 돈다.
echo "=== RE180: input_channel_re180_ref.ini ==="
exec "$HPCX/ompi/bin/mpirun" -np 2 ./build_gpu/bin/channel     apps/channel/input_channel_re180_ref.ini
