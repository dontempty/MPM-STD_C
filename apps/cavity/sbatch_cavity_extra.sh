#!/bin/bash
# BC 조합 추가 검증: lid-driven Re=100(mirror, 비영 접선 Dirichlet) + 축 교대 cavity(T Dirichlet z / Neumann x)
#SBATCH -J cavity_extra
#SBATCH -N 1
#SBATCH -n 3
#SBATCH -w cpu04
#SBATCH --time=01:00:00
#SBATCH -o /shared/home/wel1come1234/workspace/MPM-STD/apps/cavity/logs/extra_%j.log
NVR=/opt/nvidia/hpc-sdk/Linux_x86_64/23.7
HPCX=$NVR/comm_libs/12.2/hpcx/hpcx-2.15
export OPAL_PREFIX=$HPCX/ompi
export PATH=$NVR/cuda/12.2/bin:$HPCX/ompi/bin:$NVR/compilers/bin:$PATH
export LD_LIBRARY_PATH=$HPCX/ucx/lib:$HPCX/ompi/lib:$NVR/cuda/12.2/lib64:$NVR/math_libs/12.2/lib64:$NVR/compilers/lib:${LD_LIBRARY_PATH:-}
ROOT=/shared/home/wel1come1234/workspace/MPM-STD
cd $ROOT/apps/cavity
srun -n 1 -c 1 --exact $ROOT/build_cav3/bin/cavity input_lid_re100.ini > logs/lid_re100_mirror.log 2>&1 &
srun -n 1 -c 1 --exact $ROOT/build_cav3/bin/cavity input_cavity_1e3_rot.ini > logs/cavity_1e3_rot.log 2>&1 &
srun -n 1 -c 1 --exact $ROOT/build_cav3/bin/cavity input_cavity_1e4_rot.ini > logs/cavity_1e4_rot.log 2>&1 &
wait
grep -E "FINAL|error" logs/lid_re100_mirror.log logs/cavity_1e3_rot.log logs/cavity_1e4_rot.log
