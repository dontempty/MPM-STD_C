#!/bin/bash
# MPM-STD RBC NOB(글리세롤) — gpu01 (A100 40GB x 2) GPU 런.
#   sbatch -J <이름> apps/rbc_nob/sbatch_gpu01_rbc_nob.sh
#SBATCH -p batch
#SBATCH -w gpu01
#SBATCH -N 1
#SBATCH -n 2
#SBATCH --gres=gpu:2
#SBATCH -o /shared/home/wel1come1234/workspace/MPM-STD/apps/rbc_nob/log/rbc_nob_gpu_%j.log

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

# ① 스모크 (32^3, t=20, ~1분) — GPU 경로 검증 + CPU 수치 대조. 실패하면 본런 안 간다.
echo "=== SMOKE: input_rbc_nob_smoke.ini ==="
"$HPCX/ompi/bin/mpirun" -np 2 ./build_gpu/bin/rbc_nob apps/rbc_nob/input_rbc_nob_smoke.ini || { echo "SMOKE FAILED"; exit 1; }

# ② fig9 dT=40 본런 (128x128x256, t=25000, 냉시작 — 참조 P7 1:1, 수 시간)
echo "=== DT40: input_rbc_nob_dt40.ini ==="
exec "$HPCX/ompi/bin/mpirun" -np 2 ./build_gpu/bin/rbc_nob apps/rbc_nob/input_rbc_nob_dt40.ini
