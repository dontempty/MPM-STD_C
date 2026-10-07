#!/bin/bash
# LES 검증 — Vreman (2004) Phys.Fluids 16,3670 §V.B 채널 조건 1:1 (Re_tau 360, 6Hx2Hx2H, 47x63x47).
# 3종 동시: none(= 논문 0-model) / C_s 0.12(참조 Fortran) / C_s 0.17(= 논문 c=0.07).
# 각 np=7 (nz=63 = 7x9) -> 총 21 랭크. cpu 노드가 만원이라 gpu02 유휴 코어를 쓴다.
#SBATCH -J ch_les360m
#SBATCH -N 1
#SBATCH -n 14
#SBATCH --exclude=gpu01,gpu02
# LES 는 CPU 노드만 (사용자 지시 2026-10-07: GPU 는 DHVC 검증 전용)
#SBATCH --time=08:00:00
#SBATCH -o /shared/home/wel1come1234/workspace/MPM-STD/apps/channel/log/les360m_%j.log
NVR=/opt/nvidia/hpc-sdk/Linux_x86_64/23.7
HPCX=$NVR/comm_libs/12.2/hpcx/hpcx-2.15
export OPAL_PREFIX=$HPCX/ompi
export PATH=$NVR/cuda/12.2/bin:$HPCX/ompi/bin:$NVR/compilers/bin:$PATH
export LD_LIBRARY_PATH=$HPCX/ucx/lib:$HPCX/ompi/lib:$NVR/cuda/12.2/lib64:$NVR/math_libs/12.2/lib64:$NVR/compilers/lib:${LD_LIBRARY_PATH:-}
ROOT=/shared/home/wel1come1234/workspace/MPM-STD
cd $ROOT/apps/channel; mkdir -p log
for CASE in none_mirror vreman012_mirror; do
  srun -n 7 -c 1 --exact $ROOT/build/bin/channel input_channel_re360_les_${CASE}.ini \
       > log/les360_${CASE}.log 2>&1 &
done
wait
for CASE in none_mirror vreman012_mirror; do
  echo "== $CASE"; grep -E "FINAL|stats written" log/les360_${CASE}.log | tail -2
done
