#!/bin/bash
# FFT_test GPU 빌드 (glogin = aarch64, 계산 노드와 같은 아키텍처). Makefile.inc 는 고치지 않고 명령행으로 덮어쓴다
# (BUILD_HANGANG.md §3). 병렬 빌드는 사이트 규정대로 16 이하. 결과: build_gpu/bin/FFT_test
set -eu
APP=$(cd "$(dirname "$(readlink -f "$0")")" && pwd)
REPO=$(cd "${APP}/../.." && pwd)
source "${APP}/env_hangang.sh"
cd "${REPO}"
# 전치 통신 방식 시험 (transpose_transport.cpp) 용, 이 앱 링크에만 쓴다. 사이트의 NCCL 2.25.1 을 쓴다:
# aws-ofi-nccl 1.20.0 과 멀티노드 동작이 확인된 버전이다 (잡 1208206). 26.3 의 2.29.3 은 멀티노드에서 시험하지 않았다.
NCCL_HOME="/apps/NVIDIA/hpc_sdk/Linux_aarch64/25.3/comm_libs/12.8/nccl"
make GPU=1 -j"${JOBS:-9}" \
  CXX="CC --gcc-toolchain=/usr/bin/gcc-14" \
  NVHPC_DIR="${NVIDIA_PATH}" \
  CUDA_VER=12.9 \
  GPU_ARCH=cc90 \
  EXT_INC="-I${NVIDIA_PATH}/math_libs/12.9/include -I${NCCL_HOME}/include" \
  EXT_LIBS="-cudalib=cufft -L${NCCL_HOME}/lib -lnccl -Wl,-rpath,${NCCL_HOME}/lib -lm" \
  FFT_test
