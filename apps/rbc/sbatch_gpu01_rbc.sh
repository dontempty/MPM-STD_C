#!/bin/bash
# MPM-STD RBC — gpu01 (A100 40GB x 2) GPU 런.
#   sbatch -J <이름> apps/rbc/sbatch_gpu01_rbc.sh
# 환경은 Filtered_TDMA/env_gpu_local.sh 로 검증된 조합 그대로:
# hpcx-2.15 의 ucx/ompi 를 시스템 /lib UCX 1.14 보다 앞세운다.
#SBATCH -p batch
#SBATCH -w gpu01
#SBATCH -N 1
#SBATCH -n 2
#SBATCH --gres=gpu:2
#SBATCH -o /shared/home/wel1come1234/workspace/MPM-STD/apps/rbc/log/rbc_gpu_%j.log

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

# ① 스모크 (32^3, t=10, ~1분) — GPU 경로 검증 + CPU 수치 대조. 실패하면 본런 안 간다.
echo "=== SMOKE: input_rbc_smoke.ini ==="
"$HPCX/ompi/bin/mpirun" -np 2 ./build_gpu/bin/rbc apps/rbc/input_rbc_smoke.ini || { echo "SMOKE FAILED"; exit 1; }

# ② Ra=1e6 본런 (128x128x64, t=250) — 같은 할당에서 바로 이어 돈다.
echo "=== RA1E6: input_rbc_ra1e6.ini ==="
exec "$HPCX/ompi/bin/mpirun" -np 2 ./build_gpu/bin/rbc apps/rbc/input_rbc_ra1e6.ini
