#!/bin/bash
# 운동량 RHS 최적화(1·2·3) 변형 비교: K = 1 / 2 / 4 (j coarsening) 바이너리를 같은 잡에서 차례로.
#   sbatch -p <partition> --gres=gpu:N --ntasks-per-node=N --export=ALL,HW=h200,NP_LIST="1 4",CASES="strong_256 strong_512" sbatch_rhs_variants.sh
#SBATCH --time=01:30:00 --hint=nomultithread --nodes=1 --cpus-per-task=2 --mem-per-cpu=8000M
#SBATCH -J scal_rhs -o log/scal_rhs_%j.out --comment="field=mae;appl=in_house"
set -u
source "${MODULESHOME}/init/bash"; module purge; module load nvhpc/25.11_cuda12
ROOT=/scratch/r908a06/KDH/MPM-STD_C
for K in 1 2 4; do
  export BIN=$ROOT/build_gpu/bin/rbc_nob_k$K TAG=${HW}r$K REPEAT=${REPEAT:-2}
  echo "##### K=$K  BIN=$BIN TAG=$TAG"
  bash $ROOT/apps/rbc_nob/scalability/sbatch_scal_a100_1node.sh
done
