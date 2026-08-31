#!/bin/bash
#SBATCH -J dhvc_1e8_v100x8
#SBATCH --comment inhouse
#SBATCH -p cas_v100nv_8
#SBATCH --gres=gpu:8
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=4
#SBATCH -t 30:00:00
#SBATCH -o /scratch/x3319a05/MPM-STD_C/apps/dhvc/logs/dhvc_a100_1e8_v100x8.%j.log
# DHVC Ra=1e8 Pr=0.7 warm start from restart_1e7 (cross-mesh 832x416x192 -> 1536x340x768, ~401M cells), 384x192x96, np3=4 (z-slab), 4xA100 on gpu43.
# Binary: build/cuda (x86 cc80) with the new Re_inst/Re_avg monitor.csv.
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

BIN=$ROOT/build/cuda_v100/bin/apps/dhvc_gpu

# Single-runner lock for 1e8 (two candidates: from-1e6 immediate vs from-1e7
# chained; both write out_1e8/ + restart_1e8/). Loser exits 1.
LOCK=$ROOT/apps/dhvc/.lock_1e8
if ! mkdir "$LOCK" 2>/dev/null; then
  echo "another 1e8 job holds the lock — exiting"; exit 1
fi
trap 'rmdir "$LOCK"' EXIT
cd $ROOT
mkdir -p apps/dhvc/logs apps/dhvc/out_1e8

echo "=== node: $(hostname), GPUs: $CUDA_VISIBLE_DEVICES ==="
nvidia-smi -L | head -4

echo "=== DHVC Ra=1e8 warm, 4xA100, np3=4 — start $(date) ==="
mpirun --bind-to none -np 8 $BIN apps/dhvc/input_fig7_gpu_1e8_np8.toml 2>&1
echo "=== done $(date) ==="
