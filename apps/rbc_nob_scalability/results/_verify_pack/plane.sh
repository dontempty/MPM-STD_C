#!/bin/bash
#SBATCH --comment="field=mae;appl=in_house"
#SBATCH -p normal
#SBATCH -N 2
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=72
#SBATCH --gres=gpu:4
#SBATCH --exclusive
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/t525a06/KDH/MPM-STD_C/apps/rbc_nob_scalability/results/_verify_pack/plane_%j.out
source /scratch/t525a06/KDH/MPM-STD_C/apps/rbc_nob_scalability/env_kisti6.sh
B=/scratch/t525a06/KDH/MPM-STD_C/build_gpu/bin
setmode() { unset MPMSTD_PENCIL_PACK MPMSTD_PRESSURE_PENCIL; case $1 in plane_index) export MPMSTD_PENCIL_PACK=index;; t4) export MPMSTD_PRESSURE_PENCIL=transpose4;; esac; }
for g in "8 2 2 2" "4 1 2 2" "4 2 1 2" "8 4 2 1"; do
  set -- $g
  for m in plane plane_index t4; do
    setmode $m; echo "=== $2x$3x$4 $m"
    timeout 120 srun -n $1 --cpus-per-task=72 $B/test_pressure_pencil $2 $3 $4 2>&1 | grep -E 'x:|PASS|FAIL|error'
  done
done
cd /scratch/t525a06/KDH/MPM-STD_C/apps/rbc_nob_scalability/results/_verify_pack
for c in rbc_222_uneven:8 rbc_122_uneven:4 rbc_222_512:8; do
  IFS=: read name np <<< "$c"
  for m in plane t4; do
    setmode $m
    timeout 600 srun -n $np --cpus-per-task=72 $B/rbc_nob_scalability $name.ini > ${name}_$m.log 2>&1; echo "$name $m rc=$?"
  done
done
