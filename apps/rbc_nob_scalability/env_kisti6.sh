# =============================================================================
#  KISTI-6 (HANGANG, HPE Cray EX, GH200 = Grace aarch64 + Hopper sm_90) — MPM-STD GPU 환경.
#
#      source apps/rbc_nob_scalability/env_kisti6.sh
#
#  빌드(로그인 노드)와 실행(job_scal.sh 안)이 이 파일 하나를 source 한다. 조합과 이유는
#  BUILD_HANGANG.md §2·§5 (2026-10-27 검증):
#   * cray-mpich 9.1.0(기본값)의 GTL 은 libcudart.so.13 에 링크 → CUDA 12.9 와 못 섞는다. 9.0.1 로 바꾼다.
#   * 계산 노드 드라이버 570 은 CUDA 13 을 못 돌린다 → cuda/12.9_26.3.
#   * /opt/cray/pe/lib64 는 항상 9.1.0 → CRAY_LD_LIBRARY_PATH 를 앞에 둔다 (실행 때 필수).
#   * MPICH_GPU_SUPPORT_ENABLED=1 이 없으면 디바이스 포인터를 MPI 에 넘기는 순간 죽는다.
# =============================================================================
if ! type module >/dev/null 2>&1; then
    if [ -n "${MODULESHOME:-}" ] && [ -f "${MODULESHOME}/init/bash" ]; then
        source "${MODULESHOME}/init/bash"
    fi
fi

module purge >/dev/null 2>&1
module load craype-arm-grace PrgEnv-nvidia
module swap cray-mpich cray-mpich/9.0.1
module load cuda/12.9_26.3 craype-accel-nvidia90

export LD_LIBRARY_PATH=${CRAY_LD_LIBRARY_PATH}:${LD_LIBRARY_PATH:-}
export MPICH_GPU_SUPPORT_ENABLED=1
