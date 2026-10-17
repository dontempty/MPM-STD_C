#!/bin/bash
# 확장성 벤치마크 — A100-80GB ×8 1노드 (amd_a100nv_8): np 1·2·4·8 조합을 한 잡에서 순서대로.
#   cd apps/rbc_nob/scalability && sbatch sbatch_scal_a100_1node.sh
#   CASES="strong_256 weak_256" NP_LIST="1 2 4 8" REPEAT=3 로 덮어쓴다 (환경변수 --export).
#SBATCH --time=02:00:00
#SBATCH --partition=amd_a100nv_8
#SBATCH --hint=nomultithread
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=11000M
#SBATCH -J scal_a100_1n
#SBATCH -o log/scal_a100_1node_%j.out
#SBATCH --comment="field=mae;appl=in_house"
set -u
if [[ -n "${MODULESHOME:-}" && -f "${MODULESHOME}/init/bash" ]]; then source "${MODULESHOME}/init/bash"; fi
module purge
module load nvhpc/25.11_cuda12
export OMPI_MCA_opal_warn_on_missing_libcuda=0

ROOT=/scratch/r908a06/KDH/MPM-STD_C
APP=$ROOT/apps/rbc_nob
BIN=${BIN:-$ROOT/build_gpu/bin/rbc_nob}   # 전/후 비교용으로 다른 바이너리를 줄 수 있다 (예: rbc_nob_pre_buildz)
CASES=${CASES:-"strong_256 weak_256"}
NP_LIST=${NP_LIST:-"1 2 4 8"}
REPEAT=${REPEAT:-3}
TAG=${TAG:-a100}   # 하드웨어 꼬리표 — CSV 이름에 들어가 summarize.py 가 하드웨어별로 묶는다 (a100 / h200 / v100)
J=$SLURM_JOB_ID
cd "$APP"
mkdir -p scalability/log scalability/out
{
  echo "# job=$J host=$(hostname) date=$(date -Is)"
  echo "# git=$(git -C $ROOT rev-parse --short HEAD 2>/dev/null)"
  module list 2>&1
  nvidia-smi --query-gpu=name,memory.total,clocks.max.sm --format=csv
  echo "# PASCAL_TDMA_MPI_MODE=${PASCAL_TDMA_MPI_MODE:-device}"
  echo "# bin=$BIN ($(stat -c %y "$BIN"))"
} > "scalability/out/environment_${TAG}_${J}.txt"

# ① 스모크 게이트: 수치가 참조(FINAL theta_c 0.01512 / Nu 46.3863, cv12 설계노트 §3)와 같아야 측정으로 간다.
echo "=== SMOKE $(date '+%T') ==="
mpirun -np 2 --oversubscribe "$BIN" input_rbc_nob_smoke.ini > "scalability/log/smoke_${J}.log" 2>&1 || { echo "SMOKE FAILED"; tail -5 "scalability/log/smoke_${J}.log"; exit 1; }
grep "FINAL" "scalability/log/smoke_${J}.log"

# ② 조합 루프: 반복마다 로그 하나, CSV 는 out/<name>/phase_timing.csv → out/<name>/phase_timing_<job>_r<rep>.csv 로 보존.
for case in $CASES; do
  for np in $NP_LIST; do
    name="scal_${case}_np${np}"
    deck="scalability/input_${name}.ini"
    [[ -f "$deck" ]] || { echo "no deck $deck"; continue; }
    for rep in $(seq 1 $REPEAT); do
      log="scalability/log/${name}_${J}_r${rep}.log"
      echo "=== $name rep $rep np $np $(date '+%T') ==="
      ( cd scalability && mpirun -np $np "$BIN" "input_${name}.ini" > "../$log" 2>&1 )
      rc=$?
      [[ -f "scalability/out/${name}/phase_timing.csv" ]] && mv "scalability/out/${name}/phase_timing.csv" "scalability/out/${name}/phase_timing_${TAG}_${J}_r${rep}.csv"
      grep -E "^ +200 |error \(rank|blew up|step_total" "$log" | head -4
      echo "=== rc=$rc ==="
    done
  done
done
echo "# done $(date -Is)"
