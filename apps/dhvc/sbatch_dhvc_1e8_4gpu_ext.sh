#!/bin/bash
# DHVC fig7 Ra=1e8 — gpu01 + gpu02 (A100 ×4), z 분할만 (npz=4; 사용자 지시 2026-10-06).
# 확장 1e7 mirror(t=200) 격자보간 warm-start, 과도 30 + 통계 100 (2026-10-09). 2노드 mpirun 은 hpcx plm_slurm (tmp_const_gate/mpi2node 로 확인).
#SBATCH -p batch
#SBATCH -N 2
#SBATCH -w gpu01,gpu02
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:2
#SBATCH --mem=64G
#SBATCH --time=7-00:00:00
#SBATCH -o /shared/home/wel1come1234/workspace/MPM-STD/apps/dhvc/logs/dhvc_1e8_4gpu_ext_%j.log

set -u
ROOT=/shared/home/wel1come1234/workspace/MPM-STD
NVR=/opt/nvidia/hpc-sdk/Linux_x86_64/23.7
HPCX=$NVR/comm_libs/12.2/hpcx/hpcx-2.15
export OPAL_PREFIX=$HPCX/ompi
export PATH=$NVR/cuda/12.2/bin:$HPCX/ompi/bin:$NVR/compilers/bin:$PATH
export LD_LIBRARY_PATH=$HPCX/ucx/lib:$HPCX/ompi/lib:$NVR/cuda/12.2/lib64:$NVR/math_libs/12.2/lib64:$NVR/compilers/lib:${LD_LIBRARY_PATH:-}
unset PMIX_INSTALL_PREFIX
export OMPI_MCA_opal_warn_on_missing_libcuda=0
cd "$ROOT"
echo "# nodes=$SLURM_JOB_NODELIST"
exec "$HPCX/ompi/bin/mpirun" -np 4 --map-by ppr:2:node --bind-to none \
  ./build_gpu/bin/dhvc apps/dhvc/input_dhvc_1e8_4gpu_ext.ini
