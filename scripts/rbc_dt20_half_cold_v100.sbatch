#!/bin/bash
#SBATCH -J rbc_dt20_half
#SBATCH --comment="field=mae;appl=in_house"
#SBATCH -p cas_v100nv_8
#SBATCH --gres=gpu:2
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=4
#SBATCH -t 47:00:00
#SBATCH -o /scratch/x3319a05/MPM-STD_C/apps/rbc/logs/rbc_dt20_half_cold.%j.log
# dt-convergence: COLD start, identical to reorder_cold_dt20 except dt cap
# 0.001 -> 0.0005 and t_end 4000. Compare final cumulative stats (t 20->4000)
# against the parent's stats_4000000.dat (same window, dt=0.001).
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
BIN=$ROOT/build/cuda_v100/bin/apps/rbc_gpu_nob
cd $ROOT
mkdir -p apps/rbc/logs apps/rbc/reorder_cold_dt20_half apps/rbc/restart_reorder_cold_dt20_half
echo "=== node $(hostname) / RBC dt20 HALF-dt cold (cap 0.0005, 2xV100 np3=2) start $(date) ==="
mpirun --bind-to none -np 2 $BIN apps/rbc/input_reorder_cold_dt20_half.toml 2>&1 \
  | grep --line-buffered -E "step=|restart|clip|nan|Error|Nu\]"
echo "=== done $(date) ==="
