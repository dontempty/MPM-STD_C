#!/bin/bash
# 펜슬 분할 GPU 8랭크 검증 (이 클러스터, V100 ×8 한 노드): 직렬 CPU 기준(runs/1x1x1)과 30 스텝 체크포인트 비교.
#   sbatch -p cas_v100nv_8 run_compare_v100x8.sh      (CASES 로 분할 목록 덮어쓰기)
#SBATCH --time=00:30:00 --nodes=1 --ntasks-per-node=8 --gres=gpu:8 --cpus-per-task=2 --mem-per-cpu=8000M --hint=nomultithread
#SBATCH -J pencil_v100 -o log/pencil_v100x8_%j.out --comment="field=mae;appl=in_house"
set -u
if [[ -n "${MODULESHOME:-}" && -f "${MODULESHOME}/init/bash" ]]; then source "${MODULESHOME}/init/bash"; fi
module purge; module load nvhpc/25.11_cuda12
export OMPI_MCA_opal_warn_on_missing_libcuda=0
ROOT=/scratch/r908a06/KDH/MPM-STD_C
APP=$ROOT/apps/channel_pencil_test; BIN=$ROOT/build_gpu/bin/channel_pencil_test; OUT=runs_v100
cd $APP
CASES=${CASES:-"2x2x2 1x1x8 8x1x1 1x8x1 4x2x1 2x4x1 1x2x4 2x1x4 1x1x2 2x1x1 1x2x1 2x2x1"}
echo "# job=$SLURM_JOB_ID host=$(hostname) bin=$BIN ($(stat -c %y $BIN))"; nvidia-smi --query-gpu=name --format=csv,noheader | head -1
mkdir -p $OUT/1x1x1/checkpoints; \cp -f runs/1x1x1/checkpoints/*.bin $OUT/1x1x1/checkpoints/
for c in $CASES; do
  IFS=x read px py pz <<< "$c"; np=$((px*py*pz)); d=$OUT/$c
  rm -rf $d; mkdir -p $d
  sed -e "s/^npx = .*/npx = $px/" -e "s/^npy = .*/npy = $py/" -e "s/^npz = .*/npz = $pz/" \
      -e "s|^checkpoint_dir = .*|checkpoint_dir = \"checkpoints\"|" -e "s|^stats_dir = .*|stats_dir = \"stats/\"|" \
      input_base.ini > $d/input.ini
  mpirun -np $np $BIN $d/input.ini > $d/run.log 2>&1
  echo "== $c np=$np rc=$? $(grep -m1 -i error $d/run.log)"
done
echo "# 비교는 로그인 노드에서 (계산 노드 python 에 numpy 없음): python3 compare_checkpoints.py $OUT 1x1x1 $CASES"
