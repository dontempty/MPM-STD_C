#!/bin/bash
#SBATCH --job-name=rbc_mem
#SBATCH --comment="field=mae;appl=in_house"
#SBATCH --partition=normal
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=72
#SBATCH --gres=gpu:4
#SBATCH --exclusive
#SBATCH --time=00:40:00
#SBATCH --output=log/mem_%j.out
# =============================================================================
#  GPU 메모리 최대 사용량 측정 — weak 512³ / 1024³ per GPU, np 1 (slab) · np 8 (2x2x2 평면 경로).
#  런마다 nvidia-smi 를 0.2 s 간격으로 돌려 GPU 별 memory.used 최대값을 남긴다.
#  apps/rbc_nob_scalability/ 에서:  sbatch job_mem.sh    (덱: gen_decks.py --out decks_mem)
#  결과: results/_mem/<tag>.{log,csv,smi}
# =============================================================================
set -u
APP=${SLURM_SUBMIT_DIR:-$PWD}
REPO=$(cd "${APP}/../.." && pwd)
source "${APP}/env_kisti6.sh"
BIN=${REPO}/build_gpu/bin
OUT=${APP}/results/_mem
mkdir -p "${OUT}"
JOB=${SLURM_JOB_ID:-local}
export MPMSTD_MEM_REPORT=1   # 초기화 단계별 cudaMemGetInfo (main.cpp mem_mark)

run() {   # run <study> <np> <nodes>
  local st=$1 np=$2 nn=$3 tpn=$(( $2 < 4 ? $2 : 4 ))
  local tag=${st}_np${np}_j${JOB}
  # 샘플러는 같은 srun step 안에서 노드의 첫 랭크만 (별도 step 은 GPU 를 못 보고 Slingshot 설정도 깨진다)
  timeout -k 10 240 srun -N "${nn}" -n "${np}" --ntasks-per-node="${tpn}" --cpus-per-task=72 bash -c '
    if [ "${SLURM_LOCALID}" = 0 ]; then
      # 호출마다 append (nvidia-smi -l/-f 는 버퍼를 비우기 전에 kill 되면 빈 파일)
      ( while true; do nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader,nounits \
          >> "$0.$(hostname)"; sleep 0.2; done ) &
      smi=$!
    fi
    "$1" "$2" "$3"; rc=$?
    [ -n "${smi:-}" ] && { sleep 1; kill ${smi}; }
    exit ${rc}' "${OUT}/${tag}.smi" "${BIN}/rbc_nob_scalability" \
    "${APP}/decks_mem/${st}/${st}_np${np}.ini" "${OUT}/${tag}.csv" > "${OUT}/${tag}.log" 2>&1
  local rc=$?
  echo "== ${tag}  rc=${rc}"
  awk -F'[ ,]+' 'NF>=3 {k=FILENAME" gpu"$1; sub(/.*smi\./,"",k); if ($2>m[k]) m[k]=$2; t[k]=$3} END {for (k in m) printf "   %s  peak %d MiB / %d MiB\n", k, m[k], t[k]}' \
    "${OUT}/${tag}".smi.* | sort
  grep -iE "error|abort|out of memory|bad_alloc|illegal" "${OUT}/${tag}.log" | head -5 | sed 's/^/   | /'
  grep -E "^# mem " "${OUT}/${tag}.log" | sed 's/^/   /'
  grep -E "^ +[0-9]+ " "${OUT}/${tag}.log" | tail -1 | sed 's/^/   > /'
}

run weak_512_pencil 1 1
run weak_512_pencil 8 2
run weak_640_pencil 8 2
