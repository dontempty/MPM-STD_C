#!/bin/bash
#SBATCH -J cavity_1e6m
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -w cpu04
#SBATCH --time=06:00:00
#SBATCH -o /shared/home/wel1come1234/workspace/MPM-STD/apps/cavity/logs/cavity_1e6_mirror_%j.log
NVR=/opt/nvidia/hpc-sdk/Linux_x86_64/23.7
HPCX=$NVR/comm_libs/12.2/hpcx/hpcx-2.15
export OPAL_PREFIX=$HPCX/ompi
export PATH=$NVR/cuda/12.2/bin:$HPCX/ompi/bin:$NVR/compilers/bin:$PATH
export LD_LIBRARY_PATH=$HPCX/ucx/lib:$HPCX/ompi/lib:$NVR/cuda/12.2/lib64:$NVR/math_libs/12.2/lib64:$NVR/compilers/lib:${LD_LIBRARY_PATH:-}
ROOT=/shared/home/wel1come1234/workspace/MPM-STD
cd $ROOT/apps/cavity
srun -n 1 $ROOT/build/bin/cavity input_cavity_1e6_mirror.ini
