#!/bin/bash
# GPU 빌드 확인 + 펜슬 경로 GPU 실행 (2 GPU): 1x1x2(슬랩) · 2x1x1 · 1x2x1 → CPU 직렬 1x1x1 과 비교 (2026-10-20).
#SBATCH -J pencil_gpu
#SBATCH -p batch
#SBATCH -w gpu01
#SBATCH -N 1
#SBATCH -n 2
#SBATCH -c 8
#SBATCH --gres=gpu:2
#SBATCH --mem=32G
#SBATCH -t 01:00:00
#SBATCH -o /shared/home/wel1come1234/workspace/MPM-STD/apps/channel_pencil_test/log/run_compare_gpu_%j.log
set -u
ROOT=/shared/home/wel1come1234/workspace/MPM-STD
NVR=/opt/nvidia/hpc-sdk/Linux_x86_64/23.7
HPCX=$NVR/comm_libs/12.2/hpcx/hpcx-2.15
export OPAL_PREFIX=$HPCX/ompi
export PATH=$NVR/cuda/12.2/bin:$HPCX/ompi/bin:$NVR/compilers/bin:$PATH
export LD_LIBRARY_PATH=$HPCX/ucx/lib:$HPCX/ompi/lib:$NVR/cuda/12.2/lib64:$NVR/math_libs/12.2/lib64:$NVR/compilers/lib:${LD_LIBRARY_PATH:-}
unset PMIX_INSTALL_PREFIX
export OMPI_MCA_opal_warn_on_missing_libcuda=0
cd $ROOT
make GPU=1 channel_pencil_test channel -j8 > apps/channel_pencil_test/log/gpu_build_$SLURM_JOB_ID.log 2>&1 || { echo BUILD FAILED; grep -n -i " error" apps/channel_pencil_test/log/gpu_build_$SLURM_JOB_ID.log | head; exit 1; }
echo "build ok"
APP=$ROOT/apps/channel_pencil_test; cd $APP
for c in 1x1x2 2x1x1 1x2x1; do
  IFS=x read px py pz <<< "$c"; d=runs_gpu/$c; rm -rf $d; mkdir -p $d
  sed -e "s/^npx = .*/npx = $px/" -e "s/^npy = .*/npy = $py/" -e "s/^npz = .*/npz = $pz/" input_base.ini > $d/input.ini
  $HPCX/ompi/bin/mpirun -np 2 $ROOT/build_gpu/bin/channel_pencil_test $d/input.ini > $d/run.log 2>&1
  echo "== gpu $c rc=$? $(grep -E 'error' $d/run.log | tail -1)"
done
mkdir -p runs_gpu/1x1x1/checkpoints; cp runs/1x1x1/checkpoints/*.bin runs_gpu/1x1x1/checkpoints/
python3 compare_checkpoints.py runs_gpu 1x1x1 1x1x2 2x1x1 1x2x1
