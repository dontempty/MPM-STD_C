#!/bin/bash
#SBATCH --comment="field=mae;appl=in_house"
#SBATCH -p normal
#SBATCH -N 2
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=72
#SBATCH --gres=gpu:4
#SBATCH --exclusive
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/t525a06/KDH/MPM-STD_C/apps/rbc_nob_scalability/results/_verify_pack/verify_%j.out
source /scratch/t525a06/KDH/MPM-STD_C/apps/rbc_nob_scalability/env_kisti6.sh
B=/scratch/t525a06/KDH/MPM-STD_C/apps/rbc_nob_scalability/../../build_gpu/bin
cd /scratch/t525a06/KDH/MPM-STD_C/apps/rbc_nob_scalability/results/_verify_pack
for c in rbc_222_uneven:8:rbc_nob_scalability rbc_122_uneven:4:rbc_nob_scalability rbc_222_512:8:rbc_nob_scalability bl_222_wall:8:boundary_layer; do
  IFS=: read name np bin <<< "$c"
  for mode in index struct; do
    if [ $mode = index ]; then export MPMSTD_PENCIL_PACK=index; else unset MPMSTD_PENCIL_PACK; fi
    timeout 600 srun -n $np --cpus-per-task=72 $B/$bin $name.ini > ${name}_$mode.log 2>&1
    echo "$name $mode rc=$?"
  done
done
