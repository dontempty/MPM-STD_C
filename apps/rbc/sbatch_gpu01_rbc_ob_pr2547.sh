#!/bin/bash
# OB 경로 RBC(Pr 2547, 128×128×256) 냉시작 본런 — NOB 윗벽 rms 스파이크의 OB 대조 (2026-10-13), gpu01 2 GPU.
# 바이너리: build_gpu/bin/rbc (잡 36038 에서 gpu01 빌드, T/u/v rms·벽 max|U| 모니터 포함; 스모크 통과).
#SBATCH -J rbc_ob_cold
#SBATCH -p batch
#SBATCH -w gpu01
#SBATCH -N 1
#SBATCH -n 2
#SBATCH -c 4
#SBATCH --gres=gpu:2
#SBATCH --mem=64G
#SBATCH -t 12:00:00
#SBATCH -o /shared/home/wel1come1234/workspace/MPM-STD/apps/rbc/log/ob_cold_%j.log
set -u
ROOT=/shared/home/wel1come1234/workspace/MPM-STD
NVR=/opt/nvidia/hpc-sdk/Linux_x86_64/23.7
HPCX=$NVR/comm_libs/12.2/hpcx/hpcx-2.15
export OPAL_PREFIX=$HPCX/ompi
export PATH=$NVR/cuda/12.2/bin:$HPCX/ompi/bin:$NVR/compilers/bin:$PATH
export LD_LIBRARY_PATH=$HPCX/ucx/lib:$HPCX/ompi/lib:$NVR/cuda/12.2/lib64:$NVR/math_libs/12.2/lib64:$NVR/compilers/lib:${LD_LIBRARY_PATH:-}
unset PMIX_INSTALL_PREFIX
export OMPI_MCA_opal_warn_on_missing_libcuda=0
echo "# host=$(hostname) gpus=$(nvidia-smi -L | wc -l) start=$(date '+%F %T')"
cd "$ROOT/apps/rbc"
"$HPCX/ompi/bin/mpirun" -np 2 "$ROOT/build_gpu/bin/rbc" input_rbc_ob_pr2547_z256.ini
rc=$?
echo "# rc=$rc end=$(date '+%F %T')"
exit $rc
