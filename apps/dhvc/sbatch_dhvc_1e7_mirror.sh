#!/bin/bash
# DHVC fig7 Ra=1e7 mirror + 확장 통계 — gpu02 (A100 80GB x2). 1e6 restart 격자보간 warm-start.
#SBATCH -p batch
#SBATCH -w gpu02
#SBATCH --mem=64G
#SBATCH -N 1
#SBATCH -n 2
#SBATCH --gres=gpu:2
#SBATCH --time=7-00:00:00
#SBATCH -o /shared/home/wel1come1234/workspace/MPM-STD/apps/dhvc/logs/dhvc_1e7_mirror_%j.log

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
# 1e6 의 최신 체크포인트를 절대경로로 해석해 run-deck 생성
LATEST=$(ls -t "$ROOT/apps/dhvc/restart_1e6"/dhvc_ck_*.bin 2>/dev/null | head -1)
[ -z "$LATEST" ] && { echo "ERROR: restart_1e6 체크포인트 없음"; exit 1; }
echo "# warm-start 소스: $LATEST"
sed "s|WARMSTART_BIN|$LATEST|" apps/dhvc/input_dhvc_1e7_mirror.ini > apps/dhvc/input_dhvc_1e7_mirror_run.ini
echo "# host=$(hostname) gpus=$(nvidia-smi -L | wc -l)"
exec "$HPCX/ompi/bin/mpirun" -np 2 ./build_gpu/bin/dhvc apps/dhvc/input_dhvc_1e7_mirror_run.ini
