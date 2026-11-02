#!/bin/bash
# =============================================================================
#  job_scal.sh 를 노드 수별로 한꺼번에 제출한다 (apps/rbc_nob_scalability/ 에서).
#
#      bash submit_scal.sh                          # N = 1 2 4 8  (np 1..32)
#      NODES="1 2" bash submit_scal.sh              # 일부만
#      STUDIES="strong_512_slab weak_512_slab" bash submit_scal.sh
#      bash submit_scal.sh --dependency=afterok:<jobid>   # 추가 sbatch 옵션은 그대로 전달
#
#  natStr QOS 는 사용자당 실행 5 잡이 한도다 — 넘치면 먼저 경고한다.
#  STUDIES / REPS / GATE / RUN_TIMEOUT / RESULTS 는 잡으로 그대로 넘어간다 (sbatch 기본 --export=ALL).
# =============================================================================
set -u
cd "$(dirname "$(readlink -f "$0")")" || exit 1
NODES=${NODES:-"1 2 4 8"}
LIMIT=5

mkdir -p log
queued=$(squeue -h -u "${USER}" -p natStr 2>/dev/null | wc -l)
want=$(echo ${NODES} | wc -w)
if [ $(( queued + want )) -gt ${LIMIT} ]; then
    echo "WARNING: natStr 에 이미 ${queued} 잡 + 새 ${want} 잡 > 한도 ${LIMIT}. 넘치는 잡은 대기하거나 거절된다."
fi

for n in ${NODES}; do
    # 1 노드 잡은 np 1·2·4 세 개를 돌아 가장 길다. 나머지는 np 하나씩.
    t=00:40:00; [ "${n}" -eq 1 ] && t=01:30:00
    printf "N=%-3s " "${n}"
    sbatch -N "${n}" --time="${t}" --job-name="rbc_scal_N${n}" "$@" job_scal.sh
done
squeue -u "${USER}" -o "%.10i %.14j %.8T %.6D %.10l %.20R"
