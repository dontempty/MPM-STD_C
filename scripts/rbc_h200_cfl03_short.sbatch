#!/bin/bash
#SBATCH -J rbc_h200_short
#SBATCH --comment inhouse
#SBATCH -p amd_h200nv_8
#SBATCH -w gpu53
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH -t 01:00:00
#SBATCH -o /scratch/x3319a05/MPM-STD_C/apps/rbc/logs/rbc_h200_cfl03_short.%j.log
# Short restart-extraction run: RBC NOB DeltaT=40K, 256^3, cold start,
# MaxCFL=0.3, theta=1 (no MPMSTD_BLKRELAX), n_steps=1000.
# Restart: step 500 -> restart_..._short/ckpt_b/, step 1000 (end) -> restart_..._short/ .
# gpu53 = x86_64 + H200 (cc90), binary prebuilt on login node (build/cuda_h200).
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
mkdir -p apps/rbc/logs \
         apps/rbc/h200_deltat40_256_cfl03_theta1_short \
         apps/rbc/restart_h200_deltat40_256_cfl03_theta1_short

echo "=== node: $(hostname) / $(uname -m) ==="
nvidia-smi -L | head -2

echo "=== RUN start $(date) ==="
mpirun --bind-to none -np 1 $BIN apps/rbc/input_h200_deltat40_256_cfl03_theta1_short.toml 2>&1
echo "=== RUN done $(date) ==="
ls -lR apps/rbc/restart_h200_deltat40_256_cfl03_theta1_short | head -30
