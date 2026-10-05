#!/bin/bash
# de Vahl Davis 사다리 Ra 1e3~1e6 — 4 케이스를 한 할당 안에서 병렬(각 1 랭크, CPU).
#SBATCH -J cavity_ladder
#SBATCH -N 1
#SBATCH -n 4
#SBATCH -w cpu04
#SBATCH --time=06:00:00
#SBATCH -o /shared/home/wel1come1234/workspace/MPM-STD/apps/cavity/logs/ladder_%j.log
NVR=/opt/nvidia/hpc-sdk/Linux_x86_64/23.7
HPCX=$NVR/comm_libs/12.2/hpcx/hpcx-2.15
export OPAL_PREFIX=$HPCX/ompi
export PATH=$NVR/cuda/12.2/bin:$HPCX/ompi/bin:$NVR/compilers/bin:$PATH
export LD_LIBRARY_PATH=$HPCX/ucx/lib:$HPCX/ompi/lib:$NVR/cuda/12.2/lib64:$NVR/math_libs/12.2/lib64:$NVR/compilers/lib:${LD_LIBRARY_PATH:-}
ROOT=/shared/home/wel1come1234/workspace/MPM-STD
cd $ROOT/apps/cavity
for RA in 1e3 1e4 1e5 1e6; do
  srun -n 1 -c 1 --exact $ROOT/build/bin/cavity input_cavity_${RA}.ini > logs/cavity_${RA}.log 2>&1 &
done
wait
for RA in 1e3 1e4 1e5 1e6; do echo "== $RA"; tail -2 logs/cavity_${RA}.log; done
