#!/bin/bash
#SBATCH --job-name=fft_bench
#SBATCH --comment="field=mae;appl=in_house"
#SBATCH --partition=normal
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --time=00:40:00
#SBATCH --output=log/bench_%j.out
# =============================================================================
#  FFT_test 속도 측정. 제출은 apps/FFT_test/ 의 submit_bench.sh 로 한다 (멀티 노드 --exclusive 를 붙여 준다).
#  노드 수가 담당 np 를 정한다: 1 노드 = np 1 2 4, N 노드 = np N×4. NP_LIST 로 바꿀 수 있다.
#  환경변수: STUDIES      "weak_512_v2" decks/<study>/<study>_np<N>.ini 가 없는 조합은 건너뛴다
#            NP_LIST      (자동)
#            REPS         3            같은 구성 반복 — plot_events.py 가 PASS 런의 중앙값을 쓴다
#            RUN_TIMEOUT  420          런당 상한(초). 512³ 은 1 분 남짓인데 np 64 가 간헐적으로 멈춘 적이 있다 (로그의 마지막 "# stage" 가 멈춘 곳)
#  결과: results/<study>/<study>_np<N>_r<k>_j<job>_{total,phases,accuracy}.csv + .log
#  1 노드 잡은 먼저 시험 덱(decks/test)을 돌리고, 하나라도 실패하면 측정 없이 끝낸다.
#  재제출 자동화 없음. 2 런 실패하면 중단한다.
#  MPMSTD_A2A_SELF 는 설정하지 않는다: 기본값(copy)이 GTL 의 느린 self 전송을 피한다 (시작 때 값을 로그에 남긴다).
# =============================================================================
set -u
APP=${SLURM_SUBMIT_DIR:-$PWD}
[ -f "${APP}/env_hangang.sh" ] || { echo "ERROR: apps/FFT_test/ 에서 제출할 것 (지금 ${APP})"; exit 1; }
REPO=$(cd "${APP}/../.." && pwd)
source "${APP}/env_hangang.sh"
BIN=${REPO}/build_gpu/bin/FFT_test
[ -x "${BIN}" ] || { echo "ERROR: ${BIN} 없음 — apps/FFT_test/build.sh 먼저"; exit 1; }

STUDIES=${STUDIES:-"weak_512_v2"}
REPS=${REPS:-3}
RUN_TIMEOUT=${RUN_TIMEOUT:-420}
NNODES=${SLURM_JOB_NUM_NODES:-1}
JOB=${SLURM_JOB_ID:-local}
if [ -z "${NP_LIST:-}" ]; then
    if [ "${NNODES}" -eq 1 ]; then NP_LIST="1 2 4"; else NP_LIST=$(( NNODES * 4 )); fi
fi

echo "job ${JOB}  nodes ${NNODES} (${SLURM_JOB_NODELIST:-?})  np ${NP_LIST}  $(date '+%F %T')  bin $(stat -c '%y' "${BIN}" | cut -d. -f1)"
echo "studies ${STUDIES}   reps ${REPS}   git $(git -C "${REPO}" rev-parse --short HEAD 2>/dev/null)"
echo "MPICH_GPU_SUPPORT_ENABLED=${MPICH_GPU_SUPPORT_ENABLED:-} MPMSTD_A2A_SELF=${MPMSTD_A2A_SELF:-copy} PASCAL_TDMA_MPI_MODE=${PASCAL_TDMA_MPI_MODE:-device} FFT_TEST_MPI_MODE=${FFT_TEST_MPI_MODE:-device}"
module -t list 2>&1 | tr '\n' ' '; echo

if [ "${NNODES}" -eq 1 ] && [ "${RUN_TESTS:-1}" = 1 ]; then
    out=${APP}/results/test/j${JOB}
    mkdir -p "${out}"
    for deck in "${APP}"/decks/test/*.ini; do
        name=$(basename "${deck}" .ini)
        timeout -k 30 240 srun -n 4 --ntasks-per-node=4 --cpus-per-task=72 "${BIN}" "${deck}" "${out}/${name}" > "${out}/${name}.log" 2>&1
        rc=$?
        printf "test %-26s rc=%d  %s\n" "${name}" "${rc}" "$(grep -E '^(PASS|FAIL) FFT_test' "${out}/${name}.log" | tail -1)"
        grep -E "^(fft check|manufactured|random rhs|new vs old)" "${out}/${name}.log" | sed 's/^/    /'
        if [ ${rc} -ne 0 ]; then
            tail -5 "${out}/${name}.log" | sed 's/^/    | /'
            echo "!!! 시험 덱 실패 — 측정하지 않고 끝낸다"
            exit 1
        fi
    done
fi

nbad=0
for st in ${STUDIES}; do
    for np in ${NP_LIST}; do
        deck=${APP}/decks/${st}/${st}_np${np}.ini
        [ -f "${deck}" ] || { echo "skip ${st}: no deck for np ${np}"; continue; }
        mkdir -p "${APP}/results/${st}"
        tpn=$(( np < 4 ? np : 4 ))
        nodes=$(( (np + 3) / 4 ))
        for k in $(seq 1 "${REPS}"); do
            tag=${st}_np${np}_r${k}_j${JOB}
            prefix=${APP}/results/${st}/${tag}
            t0=$(date +%s)
            timeout -k 30 "${RUN_TIMEOUT}" srun -N "${nodes}" -n "${np}" --ntasks-per-node="${tpn}" --cpus-per-task=72 \
                "${BIN}" "${deck}" "${prefix}" > "${prefix}.log" 2>&1
            rc=$?
            old=$(awk -F, '$1 == "old.solve_total" {print $4}' "${prefix}_total.csv" 2>/dev/null)
            new=$(awk -F, '$1 == "new.solve_total" {print $4}' "${prefix}_total.csv" 2>/dev/null)
            verdict=$(grep -E "^(PASS|FAIL) FFT_test" "${prefix}.log" | tail -1)
            printf "%-34s rc=%d  old %10s  new %10s ms/solve  (%ds)  %s\n" "${tag%_j*}" "${rc}" "${old:-?}" "${new:-?}" "$(( $(date +%s) - t0 ))" "${verdict:-no verdict}"
            if [ ${rc} -ne 0 ] || [ -z "${new:-}" ]; then
                nbad=$((nbad + 1))
                grep -iE "error|abort|fault|FFT_test rank|FAIL" "${prefix}.log" | head -5 | sed 's/^/    | /'
                if [ ${nbad} -ge 2 ]; then echo "!!! 2 런 실패 — 중단 (원인 확인 후 다시 제출)"; exit 1; fi
            fi
        done
    done
done
echo "==== DONE $(date '+%F %T')  실패 ${nbad} 런 — plot_events.py ===="
exit ${nbad}
