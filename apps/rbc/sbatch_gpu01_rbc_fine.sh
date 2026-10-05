#!/bin/bash
# MPM-STD RBC Ra=1e6 격자 수렴 확인 런 (192x192x96) — gpu01 (A100 40GB x 2).
#   sbatch -J <이름> apps/rbc/sbatch_gpu01_rbc_fine.sh
#SBATCH -p batch
#SBATCH -w gpu01
#SBATCH -N 1
#SBATCH -n 2
#SBATCH --gres=gpu:2
#SBATCH -o /shared/home/wel1come1234/workspace/MPM-STD/apps/rbc/log/rbc_gpu_fine_%j.log

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
exec "$HPCX/ompi/bin/mpirun" -np 2 ./build_gpu/bin/rbc apps/rbc/input_rbc_ra1e6_fine.ini
