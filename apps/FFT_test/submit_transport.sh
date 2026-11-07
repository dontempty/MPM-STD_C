#!/bin/bash
# =============================================================================
#  job_transport.sh 를 노드 수별로 제출한다 (apps/FFT_test/ 에서). 잡 수 = NODES 의 개수.
#      NODES="1" bash submit_transport.sh          # 잡 1 개: np 4 (첫 시험)
#      NODES="2 4 8 16" bash submit_transport.sh   # 잡 4 개: np 8, 16, 32, 64
#  1 노드 잡이 COMPLETED + 전부 PASS 인 뒤에 나머지를 낸다. 한도를 넘으면 제출하지 않고 멈춘다 (재제출 루프 없음).
# =============================================================================
set -u
cd "$(dirname "$(readlink -f "$0")")" || exit 1
NODES=${NODES:-"1"}
LIMIT=5
mkdir -p log
queued=$(squeue -h -u "${USER}" -p normal 2>/dev/null | wc -l)
want=$(echo ${NODES} | wc -w)
if [ $(( queued + want )) -gt ${LIMIT} ]; then
    echo "STOP: normal 에 이미 ${queued} 잡 + 새 ${want} 잡 > 한도 ${LIMIT}. 제출하지 않는다."
    exit 1
fi
for n in ${NODES}; do
    extra="--exclusive"
    [ "${n}" -eq 1 ] && extra=""
    printf "N=%-3s " "${n}"
    sbatch -N "${n}" ${extra} --time=00:40:00 --job-name="fft_transport_N${n}" "$@" job_transport.sh || exit 1
done
squeue -u "${USER}" -o "%.10i %.20j %.8T %.6D %.10l %.20R"
