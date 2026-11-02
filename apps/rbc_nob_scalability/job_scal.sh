#!/bin/bash
#SBATCH --job-name=rbc_scal
#SBATCH --comment="field=mae;appl=in_house"
#SBATCH --partition=natStr
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=72
#SBATCH --gres=gpu:4
#SBATCH --exclusive
#SBATCH --time=01:00:00
#SBATCH --output=log/scal_%j.out
# =============================================================================
#  rbc_nob 확장성 측정 (strong / weak) — KISTI-6 GH200, natStr, 노드당 GPU 4 (1 랭크 = 1 GPU).
#  TDMA 백엔드는 PaScaL 만 쓴다 (기본 빌드 build_gpu/bin). apps/rbc_nob_scalability/ 에서 제출한다:
#
#      한 노드 수:          sbatch -N 2 job_scal.sh
#      노드 수별로 한꺼번에: bash submit_scal.sh
#
#  노드 수가 담당 np 를 정한다 (그 노드 수를 꽉 채우는 np 만 돈다):
#      -N 1 → np 1 2 4    -N 2 → np 8    -N 4 → np 16    -N 8 → np 32    -N 16 → np 64
#
#  환경변수 (sbatch --export=ALL,VAR=... 또는 submit_scal.sh 앞에):
#      STUDIES     "strong_256_slab weak_256_slab"   decks/<study>/<study>_np<N>.ini (gen_decks.py)
#      NP_LIST     (비우면 노드 수로 자동)            예: NP_LIST="1 2 4"
#                  항목을 np/rpn 으로 쓰면 그 런만 노드당 rpn 랭크로 펼친다.
#                  예: NP_LIST="1/1 2/1 4/1 8/4" (-N 4 잡 하나에서 np 2·4 는 노드당 1 랭크 = 노드 간 통신만)
#      RPN         4            노드당 랭크 수 기본값 (≤ 4). RPN=1 이면 -N 2 → np 2, -N 4 → np 4
#      REPS        2            같은 구성 반복 — summarize.py 는 step_total 중앙값 런을 쓴다
#      GATE        1            0 이면 rbc_nob 대조 게이트를 건너뛴다
#      RUN_TIMEOUT 900          런당 상한(초) — 하나가 걸려도 잡 전체를 태우지 않는다
#      RESULTS     results/     결과 최상위 폴더 (시험 런은 RESULTS=$PWD/results/_test)
#
#  결과: ${RESULTS}/<study>/<study>_np<N>_n<노드>x<rpn>_r<k>_j<job>.{csv,log}   (CSV = 구간 표, log = 런 출력)
#        ${RESULTS}/env/environment_j<job>.txt                    (git·모듈·GPU·노드)
#  요약: python3 summarize.py
# =============================================================================
set -u
APP=${SLURM_SUBMIT_DIR:-$PWD}
[ -f "${APP}/env_kisti6.sh" ] && [ -f "${APP}/gen_decks.py" ] \
    || { echo "ERROR: apps/rbc_nob_scalability/ 에서 제출할 것 (지금 ${APP})"; exit 1; }
REPO=$(cd "${APP}/../.." && pwd)
source "${APP}/env_kisti6.sh"

STUDIES=${STUDIES:-"strong_256_slab weak_256_slab"}
NP_ALL=${NP_ALL:-"1 2 4 8 16 32 64"}
NP_LIST=${NP_LIST:-}
REPS=${REPS:-2}
GATE=${GATE:-1}
RUN_TIMEOUT=${RUN_TIMEOUT:-900}
RESULTS=${RESULTS:-${APP}/results}

BIN=${REPO}/build_gpu/bin               # PaScaL 빌드 (make GPU=1 …; TDMA=filtered 빌드는 build_gpu_filtered/ 라 섞이지 않는다)
GPN=4                                   # GH200 노드당 GPU
RPN=${RPN:-${GPN}}                      # 노드당 랭크 기본값 (NP_LIST 항목의 /rpn 이 우선)
CPT=${SLURM_CPUS_PER_TASK:-72}          # 랭크당 Grace 코어 (노드 288 = 72 × 4)
NNODES=${SLURM_JOB_NUM_NODES:-1}
JOB=${SLURM_JOB_ID:-local}
LOGDIR=${APP}/log/j${JOB}
mkdir -p "${LOGDIR}" "${RESULTS}/env"

if [ -z "${NP_LIST}" ]; then            # 이 노드 수를 꽉 채우는 np: (NNODES-1)·RPN < np ≤ NNODES·RPN
    for np in ${NP_ALL}; do
        [ "${np}" -le $(( NNODES * RPN )) ] && [ "${np}" -gt $(( (NNODES - 1) * RPN )) ] && NP_LIST="${NP_LIST} ${np}"
    done
fi
NP_LIST=$(echo ${NP_LIST})

entry_np()  { echo "${1%/*}"; }                                   # NP_LIST 항목 "np" 또는 "np/rpn"
entry_rpn() { case $1 in */*) echo "${1#*/}" ;; *) echo "${RPN}" ;; esac; }
entry_nodes() { local np=$(entry_np "$1") r=$(entry_rpn "$1"); echo $(( (np + r - 1) / r )); }
entry_tpn() { local np=$(entry_np "$1") r=$(entry_rpn "$1"); echo $(( np < r ? np : r )); }
srun_args() {   # 항목의 np 를 노드에 rpn 씩 펼치는 srun 옵션 (timeout 이 함수를 못 부르므로 옵션만 만든다)
    echo "-N $(entry_nodes "$1") -n $(entry_np "$1") --ntasks-per-node=$(entry_tpn "$1") --cpus-per-task=${CPT}"
}

echo "############################################################"
echo " job      ${JOB}   nodes ${NNODES} (${SLURM_JOB_NODELIST:-?})   $(date '+%F %T')"
echo " np       ${NP_LIST:-<없음>}   (기본 노드당 ${RPN} 랭크; 항목 np/rpn 이 우선)"
echo " studies  ${STUDIES}"
echo " reps ${REPS}   gate ${GATE}   timeout ${RUN_TIMEOUT}s/run"
echo " results  ${RESULTS}"
echo " bin      $(stat -c '%y' "${BIN}/rbc_nob_scalability" 2>/dev/null | cut -d. -f1)  ${BIN}/rbc_nob_scalability"
echo "############################################################"

# ---- 프리플라이트: 할당을 태우기 전에 전부 확인 -------------------------------
fail=0
[ -n "${NP_LIST}" ] || { echo "ERROR: ${NNODES} 노드에 맞는 np 가 없다 (NP_LIST 로 지정)"; fail=1; }
[ -x "${BIN}/rbc_nob_scalability" ] || { echo "ERROR: ${BIN}/rbc_nob_scalability 없음 — make 먼저"; fail=1; }
[ "${GATE}" = 0 ] || [ -x "${BIN}/rbc_nob" ] || { echo "ERROR: ${BIN}/rbc_nob 없음 (게이트 대조) — make 먼저"; fail=1; }
[ "${RPN}" -ge 1 ] && [ "${RPN}" -le "${GPN}" ] || { echo "ERROR: RPN=${RPN} — 1..${GPN}"; fail=1; }
for e in ${NP_LIST}; do
    r=$(entry_rpn "${e}")
    [ "${r}" -ge 1 ] && [ "${r}" -le "${GPN}" ] || { echo "ERROR: ${e}: 노드당 랭크 ${r} — 1..${GPN}"; fail=1; continue; }
    [ "$(entry_nodes "${e}")" -le "${NNODES}" ] \
        || { echo "ERROR: ${e}: 노드 $(entry_nodes "${e}") 개 필요 > 할당 ${NNODES}"; fail=1; }
done
for st in ${STUDIES}; do
    for e in ${NP_LIST}; do
        np=$(entry_np "${e}")
        f=${APP}/decks/${st}/${st}_np${np}.ini
        [ -f "${f}" ] || echo "WARN: 덱 없음 ${f} — 이 조합은 건너뛴다 (gen_decks.py 가 만들지 않은 조합일 수 있다)"
    done
done
[ "${fail}" -eq 0 ] || { echo "프리플라이트 실패 — 중단"; exit 1; }

# ---- 환경 기록 ----------------------------------------------------------------
{
    echo "# job=${JOB} nodes=${NNODES} nodelist=${SLURM_JOB_NODELIST:-?} date=$(date -Is)"
    echo "# git=$(git -C "${REPO}" rev-parse --short HEAD 2>/dev/null) dirty_files=$(git -C "${REPO}" status --porcelain 2>/dev/null | wc -l)"
    echo "# MPICH_GPU_SUPPORT_ENABLED=${MPICH_GPU_SUPPORT_ENABLED:-} PASCAL_TDMA_MPI_MODE=${PASCAL_TDMA_MPI_MODE:-device}"
    module -t list 2>&1 | tr '\n' ' '; echo
    nvidia-smi --query-gpu=name,memory.total,driver_version,clocks.max.sm --format=csv
    nvidia-smi topo -m 2>/dev/null | head -8
    for b in rbc_nob_scalability rbc_nob; do
        [ -e "${BIN}/${b}" ] && echo "# bin ${BIN}/${b} ($(stat -c %y "${BIN}/${b}"))"
    done
} > "${RESULTS}/env/environment_j${JOB}.txt"

echo ""
echo "==== GPU / binding ===="
srun -N "${NNODES}" -n "${NNODES}" --ntasks-per-node=1 nvidia-smi -L 2>&1 | sort | uniq -c
srun -N 1 -n ${GPN} --ntasks-per-node=${GPN} --cpus-per-task="${CPT}" bash -c \
    'echo "  rank ${SLURM_PROCID} $(hostname -s) localid ${SLURM_LOCALID} cpus $(taskset -cp $$ | sed "s/.*: //")"' \
    | sort -k2 -n

# ---- 게이트: 이 잡의 가장 큰 np 로, 측정 앱 = rbc_nob 인지 (노드 간 통신까지 거친다) ---------
TOP=$(echo ${NP_LIST} | tr ' ' '\n' | sort -t/ -k1,1n | tail -1)   # 가장 큰 np 의 항목 (배치 포함)
NP_TOP=$(entry_np "${TOP}")
if [ "${GATE}" != 0 ]; then
    echo ""
    echo "==== 게이트 np=${NP_TOP} ($(entry_nodes "${TOP}")노드 x $(entry_tpn "${TOP}")): rbc_nob vs rbc_nob_scalability (decks/gate/, 60 스텝) ===="
    ref=${LOGDIR}/gate_ref_np${NP_TOP}.log new=${LOGDIR}/gate_np${NP_TOP}.log
    timeout -k 10 300 srun $(srun_args "${TOP}") "${BIN}/rbc_nob" \
        "${APP}/decks/gate/gate_ref_np${NP_TOP}.ini" > "${ref}" 2>&1
    rc_ref=$?
    timeout -k 10 300 srun $(srun_args "${TOP}") "${BIN}/rbc_nob_scalability" \
        "${APP}/decks/gate/gate_np${NP_TOP}.ini" > "${new}" 2>&1
    rc_new=$?
    if [ ${rc_ref} -ne 0 ] || [ ${rc_new} -ne 0 ]; then
        echo "  실패 rc(rbc_nob)=${rc_ref} rc(scalability)=${rc_new} — 중단"
        tail -15 "${ref}" "${new}" | sed 's/^/  | /'
        exit 1
    fi
    python3 "${APP}/check_gate.py" "${ref}" "${new}" > "${LOGDIR}/gate.txt"
    rc_gate=$?
    sed "s/^/  /" "${LOGDIR}/gate.txt"
    [ ${rc_gate} -eq 0 ] \
        || { echo "  측정 앱의 루프가 rbc_nob 과 다르다 — 측정 중단 (${ref} ${new})"; exit 1; }
fi

# ---- 측정 -------------------------------------------------------------------
NRUN=0; NBAD=0; BAD=""
run() {   # run <study> <NP_LIST 항목> <rep>
    local st=$1 e=$2 k=$3
    local np=$(entry_np "${e}")
    local deck=${APP}/decks/${st}/${st}_np${np}.ini
    [ -f "${deck}" ] || return 0
    local tag=${st}_np${np}_n$(entry_nodes "${e}")x$(entry_tpn "${e}")_r${k}_j${JOB}
    local out=${RESULTS}/${st}
    local csv=${out}/${tag}.csv log=${out}/${tag}.log
    mkdir -p "${out}"
    rm -f "${csv}"
    local t0=$(date +%s)
    timeout -k 30 "${RUN_TIMEOUT}" srun $(srun_args "${e}") "${BIN}/rbc_nob_scalability" \
        "${deck}" "${csv}" > "${log}" 2>&1
    local rc=$?
    NRUN=$((NRUN + 1))
    local total=""
    [ -f "${csv}" ] && total=$(awk -F, '$1 == "step_total" {print $4}' "${csv}")
    if [ ${rc} -ne 0 ] || [ -z "${total}" ]; then
        NBAD=$((NBAD + 1)); BAD="${BAD}\n    ${tag}  rc=${rc}  (${log})"
        printf "  %-36s FAIL rc=%s\n" "${tag%_j*}" "${rc}"
        grep -iE "error|abort|fault|illegal|blew up" "${log}" | head -5 | sed 's/^/      | /'
    else
        printf "  %-36s step_total %9.3f ms/step (rank max)   (%ds)\n" "${tag%_j*}" "${total}" $(( $(date +%s) - t0 ))
    fi
}

for st in ${STUDIES}; do
    echo ""
    echo "==== ${st} ===="
    for e in ${NP_LIST}; do
        for k in $(seq 1 "${REPS}"); do run "${st}" "${e}" "${k}"; done
    done
done

echo ""
echo "==== DONE $(date '+%F %T') ===="
if [ ${NBAD} -eq 0 ]; then
    echo "==== 전 ${NRUN} 런 정상 (rc=0, step_total 있음) — python3 summarize.py ===="
else
    echo "==== !!! ${NRUN} 런 중 ${NBAD} 런 실패 — 해당 CSV 는 쓰지 말 것 !!! ===="
    echo -e "${BAD}"
fi
exit ${NBAD}
