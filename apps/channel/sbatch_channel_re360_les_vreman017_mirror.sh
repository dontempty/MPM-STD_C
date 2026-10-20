#!/bin/bash
# Re_tau ~365 (Vreman 2004 조건) Vreman C_s 0.17 mirror 벽 판 — value 벽 판(vreman017)과 같은 조건에서 벽 처리만 바꾼 대조 (2026-10-20).
# 다른 두 mirror 런(none, 0.12)과 같은 CPU 바이너리·np 7. cpu 노드 만원이라 유휴 gpu01 의 CPU 코어 사용(GPU 미사용).
#SBATCH -J ch_les360_017m
#SBATCH -p batch
#SBATCH -w gpu01
#SBATCH -N 1
#SBATCH -n 7
#SBATCH --mem=16G
#SBATCH --time=03:00:00
#SBATCH -o /shared/home/wel1come1234/workspace/MPM-STD/apps/channel/log/les360_vreman017_mirror_job_%j.log
NVR=/opt/nvidia/hpc-sdk/Linux_x86_64/23.7
HPCX=$NVR/comm_libs/12.2/hpcx/hpcx-2.15
export OPAL_PREFIX=$HPCX/ompi
export PATH=$NVR/cuda/12.2/bin:$HPCX/ompi/bin:$NVR/compilers/bin:$PATH
export LD_LIBRARY_PATH=$HPCX/ucx/lib:$HPCX/ompi/lib:$NVR/cuda/12.2/lib64:$NVR/math_libs/12.2/lib64:$NVR/compilers/lib:${LD_LIBRARY_PATH:-}
ROOT=/shared/home/wel1come1234/workspace/MPM-STD
cd $ROOT/apps/channel
srun -n 7 -c 1 $ROOT/build/bin/channel input_channel_re360_les_vreman017_mirror.ini > log/les360_vreman017_mirror.log 2>&1
rc=$?; grep -E "FINAL|error|stats written" log/les360_vreman017_mirror.log | tail -3; echo "# rc=$rc"; exit $rc
