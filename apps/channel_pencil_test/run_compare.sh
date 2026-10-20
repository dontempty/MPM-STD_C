#!/bin/bash
# 압력 펜슬 분할 검증 — 같은 IC·30 스텝을 여러 (npx,npy,npz) 로 돌리고 마지막 체크포인트를 비교 (2026-10-20).
# 기준 = (1,1,1) 직렬과 (1,1,4) z-슬랩(기존 경로). 펜슬 = npx·npy > 1.
#SBATCH -J pencil_check
#SBATCH -p batch
#SBATCH -N 1
#SBATCH -n 8
#SBATCH --mem=16G
#SBATCH -t 00:30:00
#SBATCH -o /shared/home/wel1come1234/workspace/MPM-STD/apps/channel_pencil_test/log/run_compare_%j.log
set -u
ROOT=/shared/home/wel1come1234/workspace/MPM-STD
NVR=/opt/nvidia/hpc-sdk/Linux_x86_64/23.7
HPCX=$NVR/comm_libs/12.2/hpcx/hpcx-2.15
export OPAL_PREFIX=$HPCX/ompi
export PATH=$HPCX/ompi/bin:$PATH
export LD_LIBRARY_PATH=$HPCX/ucx/lib:$HPCX/ompi/lib:${LD_LIBRARY_PATH:-}
APP=$ROOT/apps/channel_pencil_test
cd $APP
CASES="1x1x1 1x1x4 4x1x1 1x4x1 2x2x1 2x1x2 1x2x2 2x2x2 4x2x1"
for c in $CASES; do
  IFS=x read px py pz <<< "$c"; np=$((px*py*pz))
  rm -rf runs/$c; mkdir -p runs/$c
  sed -e "s/^npx = .*/npx = $px/" -e "s/^npy = .*/npy = $py/" -e "s/^npz = .*/npz = $pz/" \
      \
      input_base.ini > runs/$c/input.ini
  srun -n $np --exact $ROOT/build/bin/channel_pencil_test runs/$c/input.ini > runs/$c/run.log 2>&1
  echo "== $c (np $np) rc=$? : $(grep -E 'error|FINAL' runs/$c/run.log | tail -1)"
done
python3 $APP/compare_checkpoints.py runs 1x1x1 $CASES
