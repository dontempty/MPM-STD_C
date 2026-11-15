#!/bin/bash
#SBATCH --comment="field=mae;appl=in_house"
#SBATCH -p normal
#SBATCH -N 2
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=72
#SBATCH --gres=gpu:4
#SBATCH --exclusive
#SBATCH --time=00:15:00
#SBATCH --output=/scratch/t525a06/KDH/MPM-STD_C/apps/rbc_nob_scalability/results/_verify_pack/tpp_%j.out
source /scratch/t525a06/KDH/MPM-STD_C/apps/rbc_nob_scalability/env_kisti6.sh
B=/scratch/t525a06/KDH/MPM-STD_C/build_gpu/bin/test_pressure_pencil
for g in "8 2 2 2" "4 1 2 2" "4 2 1 2" "8 4 2 1"; do
  set -- $g
  for mode in index struct; do
    if [ $mode = index ]; then export MPMSTD_PENCIL_PACK=index; else unset MPMSTD_PENCIL_PACK; fi
    echo "=== $2x$3x$4 $mode"
    timeout 120 srun -n $1 --cpus-per-task=72 $B $2 $3 $4 2>&1 | grep -E 'x:|PASS|FAIL|error'
  done
done
