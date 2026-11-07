#!/bin/bash
#SBATCH --job-name=fft_commtest
#SBATCH --comment="field=mae;appl=in_house"
#SBATCH --partition=normal
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --time=00:45:00
#SBATCH --output=log/commtest_%j.out
# =============================================================================
#  통신 방식 시험 (plan.md §11): 같은 데이터 패턴을 MPI_Alltoallv·p2p·pairwise 로 보내 FFTU 통신 구간을 비교한다.
#  MPM-STD 평면묶음·펜슬은 소스를 고치지 않아 항상 MPI_Alltoallv 다 — 같은 실행 안의 대조군이다.
#  노드 수가 np 를 정한다: N 노드 = np N×4. 방식은 반복마다 번갈아 돌려 시간대 변동을 고르게 한다.
#  멀티 노드라 --exclusive 필수:  sbatch -N 8 --exclusive --time=00:45:00 job_commtest.sh
#  환경변수: PROBLEMS "weak256 strong1024", MODES "alltoallv p2p pairwise", REPS 3, STEP_TIMEOUT 300, RUN_TESTS (N=8 이면 1)
#  결과: results/commtest/<문제>/np<N>/j<잡>/<방식>_r<k>*  → plot_commtest.py
#  재제출 자동화 없음. 2 스텝 실패하면 중단한다.
# =============================================================================
set -u
APP=${SLURM_SUBMIT_DIR:-$PWD}
[ -f "${APP}/env_hangang.sh" ] || { echo "ERROR: apps/FFT_test/ 에서 제출할 것 (지금 ${APP})"; exit 1; }
REPO=$(cd "${APP}/../.." && pwd)
source "${APP}/env_hangang.sh"
BIN=${REPO}/build_gpu/bin/FFT_test
[ -x "${BIN}" ] || { echo "ERROR: ${BIN} 없음"; exit 1; }

PROBLEMS=${PROBLEMS:-"weak256 strong1024"}
MODES=${MODES:-"alltoallv p2p pairwise"}
REPS=${REPS:-3}
STEP_TIMEOUT=${STEP_TIMEOUT:-300}
NNODES=${SLURM_JOB_NUM_NODES:-1}
NP=$(( NNODES * 4 ))
JOB=${SLURM_JOB_ID:-local}
RUN_TESTS=${RUN_TESTS:-$([ "${NNODES}" -eq 8 ] && echo 1 || echo 0)}

echo "job ${JOB}  nodes ${NNODES} (${SLURM_JOB_NODELIST:-?})  np ${NP}  $(date '+%F %T')  bin $(stat -c '%y' "${BIN}" | cut -d. -f1)"
echo "modes ${MODES}  problems ${PROBLEMS}  reps ${REPS}  MPMSTD_A2A_SELF=${MPMSTD_A2A_SELF:-copy}"

nbad=0
if [ "${RUN_TESTS}" = 1 ]; then   # 새 전송 경로가 맞는 해를 내는지 먼저 (1 노드 4 랭크)
    out=${APP}/results/test/j${JOB}
    mkdir -p "${out}"
    for mode in p2p pairwise; do
        for deck in "${APP}"/decks/test/t1_*.ini "${APP}"/decks/test/t3_*.ini "${APP}"/decks/test/t4_*.ini; do
            name=$(basename "${deck}" .ini)
            FFT_TEST_A2A=${mode} timeout -k 30 240 srun -N 1 -n 4 --ntasks-per-node=4 --cpus-per-task=72 "${BIN}" "${deck}" "${out}/${mode}_${name}" > "${out}/${mode}_${name}.log" 2>&1
            rc=$?
            printf "test %-9s %-22s rc=%d  %s\n" "${mode}" "${name}" "${rc}" "$(grep -E '^(PASS|FAIL) FFT_test' "${out}/${mode}_${name}.log" | tail -1)"
            if [ ${rc} -ne 0 ]; then tail -5 "${out}/${mode}_${name}.log" | sed 's/^/    | /'; echo "!!! 시험 실패 — 측정하지 않고 끝낸다"; exit 1; fi
        done
    done
fi

for problem in ${PROBLEMS}; do
    deck=${APP}/decks/compare_${problem}/compare_${problem}_np${NP}.ini
    [ -f "${deck}" ] || { echo "skip ${problem}: no deck for np ${NP}"; continue; }
    base=${APP}/results/commtest/${problem}/np${NP}/j${JOB}
    mkdir -p "${base}"
    echo "==== ${problem} np ${NP}  $(date '+%T') ===="
    for k in $(seq 1 "${REPS}"); do
        for mode in ${MODES}; do
            prefix=${base}/${mode}_r${k}
            FFT_TEST_A2A=${mode} timeout -k 30 "${STEP_TIMEOUT}" srun -N "${NNODES}" -n "${NP}" --ntasks-per-node=4 --cpus-per-task=72 --kill-on-bad-exit=1 "${BIN}" "${deck}" "${prefix}" > "${prefix}.log" 2>&1
            rc=$?
            if [ ${rc} -ne 0 ] || ! grep -q "^PASS FFT_test" "${prefix}.log"; then
                nbad=$((nbad + 1))
                echo "  FAIL ${mode} r${k} rc=${rc}  last stage: $(grep -E '^# stage' "${prefix}.log" | tail -1 | cut -c1-60)"
                grep -iE "error|abort|fault|FFT_test rank" "${prefix}.log" | head -5 | sed 's/^/    | /'
                if [ ${nbad} -ge 2 ]; then echo "!!! 2 스텝 실패 — 중단 (원인 확인 후 다시 제출)"; exit 1; fi
                continue
            fi
            printf "  r%d %-10s %s\n" "${k}" "${mode}" "$(awk -F, '!/^#/ && NR>1 && $1!="method" {printf "%s %.2f  ", $1, $2}' "${prefix}_times.csv")"
        done
    done
done
echo "==== DONE $(date '+%F %T')  실패 ${nbad} 스텝 ===="
exit ${nbad}
