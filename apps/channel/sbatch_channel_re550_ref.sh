#!/bin/bash
# MPM-STD channel Re_tau=550 참조격자(768x512x384) — GPU 노드 (gpu01 40GB / gpu02 80GB 어느 쪽이든).
#   sbatch -J <이름> apps/channel/sbatch_channel_re550_ref.sh
#SBATCH -p batch
#SBATCH -N 1
#SBATCH -n 2
#SBATCH --gres=gpu:2
#SBATCH -o /shared/home/wel1come1234/workspace/MPM-STD/apps/channel/log/channel_re550_ref_gpu_%j.log

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

# ① 스모크 (32x24x32, ~1분) — gpu02 에서의 GPU 경로 검증. 실패하면 본런 안 간다.
echo "=== SMOKE: input_channel_smoke.ini ==="
"$HPCX/ompi/bin/mpirun" -np 2 ./build_gpu/bin/channel apps/channel/input_channel_smoke_budget.ini || { echo "SMOKE FAILED"; exit 1; }

# ② Re_tau 550 참조격자 본런 (768x512x384, 120000 스텝, ~11시간)
echo "=== RE550: input_channel_re550_ref.ini ==="
exec "$HPCX/ompi/bin/mpirun" -np 2 ./build_gpu/bin/channel apps/channel/input_channel_re550_ref.ini
