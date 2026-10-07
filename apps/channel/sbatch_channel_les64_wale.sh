#!/bin/bash
# LES WALE 검증 — 64³ mirror, np=4 (gpu02 유휴 코어). 비교 짝: none_mirror · vreman012_mirror (잡 35722)
#SBATCH -J ch_les64w
#SBATCH -N 1
#SBATCH -n 4
#SBATCH --exclude=gpu01,gpu02
# LES 는 CPU 노드만 (사용자 지시 2026-10-07: GPU 는 DHVC 검증 전용)
#SBATCH --time=06:00:00
#SBATCH -o /shared/home/wel1come1234/workspace/MPM-STD/apps/channel/log/les64w_%j.log
NVR=/opt/nvidia/hpc-sdk/Linux_x86_64/23.7
HPCX=$NVR/comm_libs/12.2/hpcx/hpcx-2.15
export OPAL_PREFIX=$HPCX/ompi
export PATH=$NVR/cuda/12.2/bin:$HPCX/ompi/bin:$NVR/compilers/bin:$PATH
export LD_LIBRARY_PATH=$HPCX/ucx/lib:$HPCX/ompi/lib:$NVR/cuda/12.2/lib64:$NVR/math_libs/12.2/lib64:$NVR/compilers/lib:${LD_LIBRARY_PATH:-}
ROOT=/shared/home/wel1come1234/workspace/MPM-STD
cd $ROOT/apps/channel; mkdir -p log
for CASE in wale_mirror; do
  srun -n 4 -c 1 --exact $ROOT/build/bin/channel input_channel_re180_les64_${CASE}.ini > log/les64_${CASE}.log 2>&1 &
done
wait
for CASE in wale_mirror; do echo "== $CASE"; grep -E "FINAL|stats written" log/les64_${CASE}.log | tail -2; done
