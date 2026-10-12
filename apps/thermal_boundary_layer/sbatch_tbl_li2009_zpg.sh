#!/bin/bash
# 열 경계층 검증 (Li et al. 2009 조건) — gpu02 2 GPU (2026-10-11)
#SBATCH -J tbl_li2009_zpg
#SBATCH -p batch
#SBATCH -w gpu02
#SBATCH -N 1
#SBATCH -n 2
#SBATCH --gres=gpu:2
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH -o /shared/home/wel1come1234/workspace/MPM-STD/apps/thermal_boundary_layer/logs/tbl_li2009_zpg_%j.log
set -u
ROOT=/shared/home/wel1come1234/workspace/MPM-STD
NVR=/opt/nvidia/hpc-sdk/Linux_x86_64/23.7
HPCX=$NVR/comm_libs/12.2/hpcx/hpcx-2.15
export OPAL_PREFIX=$HPCX/ompi
export PATH=$NVR/cuda/12.2/bin:$HPCX/ompi/bin:$NVR/compilers/bin:$PATH
export LD_LIBRARY_PATH=$HPCX/ucx/lib:$HPCX/ompi/lib:$NVR/cuda/12.2/lib64:$NVR/math_libs/12.2/lib64:$NVR/compilers/lib:${LD_LIBRARY_PATH:-}
unset PMIX_INSTALL_PREFIX
export OMPI_MCA_opal_warn_on_missing_libcuda=0
cd $ROOT/apps/thermal_boundary_layer
"$HPCX/ompi/bin/mpirun" -np 2 $ROOT/build_gpu/bin/thermal_boundary_layer input_tbl_li2009_zpg.ini
