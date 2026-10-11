#!/bin/bash
# 경계층 LES 격자 수렴 (1024x96x128) — gpu01 2 GPU: vreman017_fine (2026-10-11)
#SBATCH -J bl_les_vreman017_fine017_fine
#SBATCH -p batch
#SBATCH -w gpu01
#SBATCH -N 1
#SBATCH -n 2
#SBATCH --gres=gpu:2
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH -o /shared/home/wel1come1234/workspace/MPM-STD/apps/boundary_layer/logs/bl_les_vreman017_fine_%j.log
set -u
ROOT=/shared/home/wel1come1234/workspace/MPM-STD
NVR=/opt/nvidia/hpc-sdk/Linux_x86_64/23.7
HPCX=$NVR/comm_libs/12.2/hpcx/hpcx-2.15
export OPAL_PREFIX=$HPCX/ompi
export PATH=$NVR/cuda/12.2/bin:$HPCX/ompi/bin:$NVR/compilers/bin:$PATH
export LD_LIBRARY_PATH=$HPCX/ucx/lib:$HPCX/ompi/lib:$NVR/cuda/12.2/lib64:$NVR/math_libs/12.2/lib64:$NVR/compilers/lib:${LD_LIBRARY_PATH:-}
unset PMIX_INSTALL_PREFIX
export OMPI_MCA_opal_warn_on_missing_libcuda=0
cd $ROOT/apps/boundary_layer
"$HPCX/ompi/bin/mpirun" -np 2 $ROOT/build_gpu/bin/boundary_layer input_bl_re4500_les_vreman017_fine.ini > logs/bl_re4500_les_vreman017_fine.log 2>&1; echo "== vreman017_fine: $(grep FINAL logs/bl_re4500_les_vreman017_fine.log)"
