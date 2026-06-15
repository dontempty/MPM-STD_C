#!/bin/bash
#SBATCH -p batch
#SBATCH -w gpu01
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --gres=gpu:a100:2
#SBATCH -o apps/dhvc/logs/dhvc_gpu_1e6.log
#SBATCH --time=08:00:00
cd /shared/home/wel1come1234/workspace/MPM-STD_C
source /etc/profile.d/modules.sh; source /etc/profile.d/z95_nvhpc_modules.sh
module purge
module load nvhpc-nompi/23.7
module use /opt/nvidia/hpc-sdk/Linux_x86_64/23.7/comm_libs/12.2/hpcx/latest/modulefiles
module load hpcx
echo "host=$(hostname) Ra=1e6 $(date)"
export OMPI_CXX=nvc++
mkdir -p apps/dhvc/out_1e6
srun --mpi=pmix_v3 -n 2 build/cuda/bin/apps/dhvc_gpu apps/dhvc/input_fig7_gpu_1e6.toml
echo "DONE Ra=1e6 $(date)"
