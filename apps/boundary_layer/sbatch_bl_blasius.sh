#!/bin/bash
# 단계 A: 2D Blasius 경계층 (CPU 1 랭크)
#SBATCH -J bl_blasius
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -w cpu04
#SBATCH --time=01:00:00
#SBATCH -o /shared/home/wel1come1234/workspace/MPM-STD/apps/boundary_layer/logs/blasius_%j.log
NVR=/opt/nvidia/hpc-sdk/Linux_x86_64/23.7
HPCX=$NVR/comm_libs/12.2/hpcx/hpcx-2.15
export OPAL_PREFIX=$HPCX/ompi
export PATH=$NVR/cuda/12.2/bin:$HPCX/ompi/bin:$NVR/compilers/bin:$PATH
export LD_LIBRARY_PATH=$HPCX/ucx/lib:$HPCX/ompi/lib:$NVR/cuda/12.2/lib64:$NVR/math_libs/12.2/lib64:$NVR/compilers/lib:${LD_LIBRARY_PATH:-}
ROOT=/shared/home/wel1come1234/workspace/MPM-STD
cd $ROOT/apps/boundary_layer
srun -n 1 $ROOT/build/bin/boundary_layer ${1:-input_bl_blasius.ini}
