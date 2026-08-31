#!/bin/bash
#SBATCH -J rbc_dt40_h200
#SBATCH --comment inhouse
#SBATCH -p amd_h200nv_8
#SBATCH -w gpu53
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH -t 24:00:00
#SBATCH -o /scratch/x3319a05/MPM-STD_C/apps/rbc/logs/rbc_h200_deltat40_256_cfl03_theta1.%j.log
# ---------------------------------------------------------------------------
# RBC NOB DeltaT=40K, 256^3, cold start, MaxCFL=0.3, theta=1 (no MPMSTD_BLKRELAX),
# n_steps=100000. Same as the gpu51 (GH200) run except CFL 1.0 -> 0.3.
# Restart: step 50000 -> restart_.../ckpt_b/, end of run -> restart_.../ .
#
# gpu53 is x86_64 + H200 (cc90): binary prebuilt on the login node with
# nvhpc/25.11_cuda12 into build/cuda_h200 (default rdc; the nordc workaround
# was only needed for the aarch64 24.5 toolchain). A 200-step smoke runs first
# in case the rdc registration bug also bites this stack.
# ---------------------------------------------------------------------------
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
         apps/rbc/h200_smoke apps/rbc/restart_h200_smoke \
         apps/rbc/h200_deltat40_256_cfl03_theta1 \
         apps/rbc/restart_h200_deltat40_256_cfl03_theta1

echo "=== node: $(hostname) / $(uname -m) ==="
nvidia-smi -L | head -2

echo "=== SMOKE (200 steps) start $(date) ==="
timeout 900 mpirun --bind-to none -np 1 $BIN apps/rbc/input_h200_smoke.toml 2>&1 | tail -8
echo "=== SMOKE ok — PRODUCTION start $(date) ==="
mpirun --bind-to none -np 1 $BIN apps/rbc/input_h200_deltat40_256_cfl03_theta1.toml 2>&1
echo "=== RUN done $(date) ==="
ls -lR apps/rbc/restart_h200_deltat40_256_cfl03_theta1 | head -30
