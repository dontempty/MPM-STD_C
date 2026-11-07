#!/bin/bash
# =============================================================================
#  job_bench.sh 를 노드 수별로 제출한다 (apps/FFT_test/ 에서). 잡 수 = NODES 의 개수.
#      NODES="1 2 4 8 16" bash submit_bench.sh     # np 4 8 16 32 64 — 5 잡 (사용자당 파티션 한도 5)
#  시험 잡(job_test.sh)이 COMPLETED + 전부 PASS 인 뒤에만 쓴다. 넘치면 제출하지 않고 멈춘다 (재제출 루프 없음).
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
    # 1 노드 잡은 np 1·2·4 세 개를 돌아 가장 길다. 멀티 노드는 np 하나에 --exclusive 필수.
    extra="--exclusive"; t=00:40:00
    [ "${n}" -eq 1 ] && { extra=""; t=01:30:00; }
    printf "N=%-3s " "${n}"
    sbatch -N "${n}" ${extra} --time="${t}" --job-name="fft_bench_N${n}" "$@" job_bench.sh || exit 1
done
squeue -u "${USER}" -o "%.10i %.16j %.8T %.6D %.10l %.20R"
