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
# OLD_ROOT 가 있으면 그 트리(예: 펜슬 이전 커밋 워크트리)도 GPU 빌드해 슬랩 1x1x2 를 비트 비교한다.
APP=$ROOT/apps/channel_pencil_test
OLD_ROOT=${OLD_ROOT:-}
OUT=${OUT:-runs_gpu}
CPU_REF=${CPU_REF:-runs}          # CPU 직렬 1x1x1 체크포인트 폴더 (APP 기준)
for TREE in $ROOT $OLD_ROOT; do
  (cd $TREE && make GPU=1 channel_pencil_test -j8 > $APP/log/gpu_build_$(basename $TREE)_$SLURM_JOB_ID.log 2>&1) \
    || { echo "BUILD FAILED in $TREE"; grep -n -i " error" $APP/log/gpu_build_$(basename $TREE)_$SLURM_JOB_ID.log | head; exit 1; }
  echo "build ok: $TREE"
done
cd $APP
run_gpu() {  # run_gpu <tree> <case> <outdir>
  IFS=x read px py pz <<< "$2"; d=$3/$2; rm -rf $d; mkdir -p $d
  sed -e "s/^npx = .*/npx = $px/" -e "s/^npy = .*/npy = $py/" -e "s/^npz = .*/npz = $pz/" input_base.ini > $d/input.ini
  $HPCX/ompi/bin/mpirun -np 2 $1/build_gpu/bin/channel_pencil_test $d/input.ini > $d/run.log 2>&1
  echo "== gpu $2 ($1) rc=$? $(grep -E 'error' $d/run.log | tail -1)"
}
for c in 1x1x2 2x1x1 1x2x1; do run_gpu $ROOT $c $OUT; done
mkdir -p $OUT/1x1x1/checkpoints; cp $CPU_REF/1x1x1/checkpoints/*.bin $OUT/1x1x1/checkpoints/
python3 compare_checkpoints.py $OUT 1x1x1 1x1x2 2x1x1 1x2x1
if [ -n "$OLD_ROOT" ]; then
  run_gpu $OLD_ROOT 1x1x2 ${OUT}_old
  cmp ${OUT}_old/1x1x2/checkpoints/*.bin $OUT/1x1x2/checkpoints/*.bin && echo "GPU slab 1x1x2: pre-pencil vs new BIT-IDENTICAL" || echo "GPU slab 1x1x2: DIFFERS"
fi
