#!/bin/bash
# channel_ftdma 통계 런 (2026-10-28): 256³ DNS Re_τ 180, 2x2x2 = 2 노드 × GH200 4장, 랭크당 GPU 1장.
#   sbatch -J ftdma_pascal   --export=ALL,TAG=dns256_pascal_2x2x2   sbatch_stat.sh
#   sbatch -J ftdma_filtered --export=ALL,TAG=dns256_filtered_2x2x2 sbatch_stat.sh
# TAG = dns256_<backend>_<분할> → 덱 input_$TAG.ini, 출력 stats_$TAG/ (통계 + truncation_J.dat), checkpoints_$TAG/, log/$TAG.log
#SBATCH -p natStr
#SBATCH -N 2
#SBATCH --ntasks-per-node=4
#SBATCH --exclusive
#SBATCH --gres=gpu:4
#SBATCH --time=1-00:00:00
#SBATCH --comment="field=mae;appl=in_house"
#SBATCH -o /scratch/t639a05/MPM-STD_C/apps/channel_ftdma/log/%x_%j.out
set -u
module purge
module load craype-arm-grace PrgEnv-nvidia
module swap cray-mpich cray-mpich/9.0.1
module load cuda/12.9_26.3 craype-accel-nvidia90
export LD_LIBRARY_PATH=$CRAY_LD_LIBRARY_PATH:$LD_LIBRARY_PATH
export MPICH_GPU_SUPPORT_ENABLED=1
ROOT=/scratch/t639a05/MPM-STD_C
cd $ROOT/apps/channel_ftdma
b=$(echo $TAG | cut -d_ -f2); d=${TAG##*_}; IFS=x read px py pz <<< "$d"; np=$((px*py*pz))
case $b in pascal) BIN=$ROOT/build_gpu/bin/channel_ftdma ;; filtered) BIN=$ROOT/build_gpu_filtered/bin/channel_ftdma ;; *) echo "bad TAG $TAG"; exit 1 ;; esac
/bin/rm -rf stats_$TAG checkpoints_$TAG
echo "# $TAG backend=$b np=$np nodes=$SLURM_JOB_NODELIST bin=$BIN ($(stat -c %y $BIN)) start=$(date '+%F %T')"
srun -n $np "$BIN" input_$TAG.ini > log/$TAG.log 2>&1
echo "# rc=$? end=$(date '+%F %T')  $(grep FINAL log/$TAG.log)"
