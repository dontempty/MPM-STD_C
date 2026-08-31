#!/bin/bash
#SBATCH -J rbc_dtsmall_long
#SBATCH --comment inhouse
#SBATCH -p amd_h200nv_8
#SBATCH -w gpu53
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH -t 01:00:00
#SBATCH -o /scratch/x3319a05/MPM-STD_C/apps/rbc/logs/rbc_h200_checker_exp5.%j.log
# Fixed-dt checkerboard experiments (vs baseline dT=40/theta=1/CFL-dt~0.0031):
#   exp3: dt pinned 0.0015 (cap binds) -> D_top~0.65 subcritical: top checkerboard should vanish
#   exp4: dt pinned 0.006  (MaxCFL=1 so cap binds) -> D_top~2.6: top persists, bottom stays clean
set -eo pipefail

ROOT=/scratch/x3319a05/MPM-STD_C
NVR=/apps/compiler/nvidia_hpc_sdk/25.11/Linux_x86_64/25.11
FFTW=/apps/libraries/fftw3/3.3.10/gcc/11.5.0
export PATH=$NVR/cuda/12.9/bin:$NVR/comm_libs/12.9/hpcx/hpcx-2.25.1/ompi/bin:$NVR/compilers/bin:$PATH
export LD_LIBRARY_PATH=$NVR/cuda/12.9/lib64:$NVR/comm_libs/12.9/hpcx/hpcx-2.25.1/ompi/lib:$NVR/math_libs/12.9/lib64:$NVR/compilers/lib:$FFTW/lib:$LD_LIBRARY_PATH
unset PMIX_INSTALL_PREFIX
export OMPI_MCA_opal_warn_on_missing_libcuda=0
export UCX_TLS=cuda_copy,cuda_ipc,sm,self
export UCX_MEMTYPE_CACHE=n

BIN=$ROOT/build/cuda_h200/bin/apps/rbc_gpu_nob
cd $ROOT
mkdir -p apps/rbc/h200_exp3_dtsmall apps/rbc/restart_h200_exp3_dtsmall \
         apps/rbc/h200_exp4_dtlarge apps/rbc/restart_h200_exp4_dtlarge

mkdir -p apps/rbc/h200_exp5_dtsmall_long apps/rbc/restart_h200_exp5_dtsmall_long
echo "=== EXP5 (dt cap 0.0015, 10000 steps -> t~15) start $(date) ==="
mpirun --bind-to none -np 1 $BIN apps/rbc/input_h200_exp5_dtsmall_long.toml 2>&1
echo "=== done $(date) ==="
