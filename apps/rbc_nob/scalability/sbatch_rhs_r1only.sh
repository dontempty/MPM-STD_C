#!/bin/bash
# 운동량 RHS 최적화 1(역수)만 적용된 실험 바이너리 측정 — 3(융합) 기여 분리용
#SBATCH --time=00:40:00 --hint=nomultithread --nodes=1 --cpus-per-task=2 --mem-per-cpu=8000M
#SBATCH -J scal_r1only -o log/scal_r1only_%j.out --comment="field=mae;appl=in_house"
set -u
source "${MODULESHOME}/init/bash"; module purge; module load nvhpc/25.11_cuda12
ROOT=/scratch/r908a06/KDH/MPM-STD_C
export BIN=$ROOT/build_gpu/bin/rbc_nob_r1only TAG=${HW}r1only REPEAT=${REPEAT:-2}
bash $ROOT/apps/rbc_nob/scalability/sbatch_scal_a100_1node.sh
