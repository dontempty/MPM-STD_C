#!/bin/bash
#SBATCH -J rbc_cold_dt20
#SBATCH --comment inhouse
#SBATCH -p amd_a100_4
#SBATCH -w gpu45
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH -t 47:00:00
#SBATCH -o /scratch/x3319a05/MPM-STD_C/apps/rbc/logs/rbc_reorder_cold_dt20.%j.log
# RBC NOB DeltaT=20, 128^3, cold start, MaxCFL=0.3 + dt cap 0.001 (checkerboard-
# subcritical: D~0.05 at the cold wall), n_steps=8M, periodic ckpt every 100k.
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

BIN=$ROOT/build/cuda/bin/apps/rbc_gpu_nob
cd $ROOT
mkdir -p apps/rbc/logs apps/rbc/reorder_cold_dt20 apps/rbc/restart_reorder_cold_dt20

echo "=== node: $(hostname), GPU: $CUDA_VISIBLE_DEVICES ==="
nvidia-smi -L | head -1
echo "=== RBC reorder_cold_dt20 start $(date) ==="
mpirun --bind-to none -np 1 $BIN apps/rbc/input_reorder_cold_dt20.toml 2>&1 \
  | grep --line-buffered -E "step=|restart|ckpt|clip|nan|NaN|error|Error|Nu\]|TIMING"
echo "=== done $(date) ==="
