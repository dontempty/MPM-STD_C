#!/bin/bash
# LES 96³ 격자 민감도 2종(mirror 벽, 각 np=16): none / constant_vreman C_s 0.12 — 32 코어 비는 노드 아무 곳
#SBATCH -J ch_les96m
#SBATCH -N 1
#SBATCH -n 32
# 노드 미지정: 32 코어가 비는 아무 노드 (cpu04 는 상위 우선순위 잡이 선점 중, 2026-10-07)
#SBATCH --time=12:00:00
#SBATCH -o /shared/home/wel1come1234/workspace/MPM-STD/apps/channel/log/les96m_%j.log
NVR=/opt/nvidia/hpc-sdk/Linux_x86_64/23.7
HPCX=$NVR/comm_libs/12.2/hpcx/hpcx-2.15
export OPAL_PREFIX=$HPCX/ompi
export PATH=$NVR/cuda/12.2/bin:$HPCX/ompi/bin:$NVR/compilers/bin:$PATH
export LD_LIBRARY_PATH=$HPCX/ucx/lib:$HPCX/ompi/lib:$NVR/cuda/12.2/lib64:$NVR/math_libs/12.2/lib64:$NVR/compilers/lib:${LD_LIBRARY_PATH:-}
ROOT=/shared/home/wel1come1234/workspace/MPM-STD
cd $ROOT/apps/channel; mkdir -p log
for CASE in none_mirror vreman012_mirror; do
  srun -n 16 -c 1 --exact $ROOT/build/bin/channel input_channel_re180_les96_${CASE}.ini > log/les96_${CASE}.log 2>&1 &
done
wait
for CASE in none_mirror vreman012_mirror; do echo "== $CASE"; grep -E "FINAL|stats written" log/les96_${CASE}.log | tail -2; done
