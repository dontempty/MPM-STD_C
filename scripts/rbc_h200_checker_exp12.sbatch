#!/bin/bash
#SBATCH -J rbc_checker_exp
#SBATCH --comment inhouse
#SBATCH -p amd_h200nv_8
#SBATCH -w gpu53
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH -t 01:00:00
#SBATCH -o /scratch/x3319a05/MPM-STD_C/apps/rbc/logs/rbc_h200_checker_exp12.%j.log
# Checkerboard attribution experiments (vs baseline restart_h200_..._short,
# which is DeltaT=40 / theta=1 / CFL=0.3 / 1000 steps):
#   exp1: DeltaT=0.01  (mu~1 everywhere -> if top checkerboard vanishes, NOB mu is confirmed)
#   exp2: DeltaT=40 + MPMSTD_BLKRELAX=0.5 (if it vanishes, block-coupling feed-forward
#         path is the carrier; if it stays, the explicit RHS transpose stress is)
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
mkdir -p apps/rbc/h200_exp1_deltat001 apps/rbc/restart_h200_exp1_deltat001 \
         apps/rbc/h200_exp2_dt40_blk05 apps/rbc/restart_h200_exp2_dt40_blk05

echo "=== EXP1 (DeltaT=0.01, theta=1) start $(date) ==="
mpirun --bind-to none -np 1 $BIN apps/rbc/input_h200_exp1_deltat001.toml 2>&1 | tail -15

echo "=== EXP2 (DeltaT=40, MPMSTD_BLKRELAX=0.5) start $(date) ==="
mpirun --bind-to none -np 1 -x MPMSTD_BLKRELAX=0.5 $BIN apps/rbc/input_h200_exp2_dt40_blk05.toml 2>&1 | tail -15

echo "=== both done $(date) ==="
ls -l apps/rbc/restart_h200_exp1_deltat001 apps/rbc/restart_h200_exp2_dt40_blk05
