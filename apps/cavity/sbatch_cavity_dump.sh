#!/bin/bash
# 완주한 케이스를 최종 체크포인트에서 2 시간단위만 더 돌려 2D 장 덤프를 뽑는다 (벤치마크 후처리용)
#SBATCH -J cavity_dump
#SBATCH -N 1
#SBATCH -n 3
#SBATCH -w cpu04
#SBATCH --time=01:00:00
#SBATCH -o /shared/home/wel1come1234/workspace/MPM-STD/apps/cavity/logs/dump_%j.log
NVR=/opt/nvidia/hpc-sdk/Linux_x86_64/23.7
HPCX=$NVR/comm_libs/12.2/hpcx/hpcx-2.15
export OPAL_PREFIX=$HPCX/ompi
export PATH=$NVR/cuda/12.2/bin:$HPCX/ompi/bin:$NVR/compilers/bin:$PATH
export LD_LIBRARY_PATH=$HPCX/ucx/lib:$HPCX/ompi/lib:$NVR/cuda/12.2/lib64:$NVR/math_libs/12.2/lib64:$NVR/compilers/lib:${LD_LIBRARY_PATH:-}
ROOT=/shared/home/wel1come1234/workspace/MPM-STD
cd $ROOT/apps/cavity
for RA in 1e3 1e4 1e5; do
  srun -n 1 -c 1 --exact $ROOT/build_cav2/bin/cavity input_cavity_${RA}_dump.ini > logs/cavity_${RA}_dump.log 2>&1 &
done
wait
for RA in 1e3 1e4 1e5; do echo "== $RA"; grep -E "FINAL|field dump" logs/cavity_${RA}_dump.log; done
