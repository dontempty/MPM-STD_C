#!/bin/bash
#SBATCH --comment="field=mae;appl=in_house"
#SBATCH -p normal
#SBATCH -N 16
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=72
#SBATCH --gres=gpu:4
#SBATCH --exclusive
#SBATCH --time=00:15:00
#SBATCH --output=/scratch/t525a06/KDH/MPM-STD_C/apps/rbc_nob_scalability/results/_diag/diag_%j.out
source /scratch/t525a06/KDH/MPM-STD_C/apps/rbc_nob_scalability/env_kisti6.sh
BIN=/scratch/t525a06/KDH/MPM-STD_C/apps/rbc_nob_scalability/../../build_gpu/bin/rbc_nob_scalability
for c in small mid big; do
  echo "=== $c"; srun -n 64 --cpus-per-task=72 -l $BIN /scratch/t525a06/KDH/MPM-STD_C/apps/rbc_nob_scalability/results/_diag/$c.ini /scratch/t525a06/KDH/MPM-STD_C/apps/rbc_nob_scalability/results/_diag/$c.csv > /scratch/t525a06/KDH/MPM-STD_C/apps/rbc_nob_scalability/results/_diag/$c.log 2> /scratch/t525a06/KDH/MPM-STD_C/apps/rbc_nob_scalability/results/_diag/$c.err; echo rc=$?
done
