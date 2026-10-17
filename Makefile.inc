# ══════════════════════════════════════════════════════════════════════════
# 빌드 설정 — 컴파일러·옵션·외부 라이브러리 경로는 전부 이 파일에서 정한다.
# Makefile 은 규칙만 갖고 이 파일을 include 한다. 머신이 바뀌면 여기만 고친다.
# (MPM-STD_C 의 Makefile.inc 와 같은 역할 — 2026-10-04 사용자 결정)
# ══════════════════════════════════════════════════════════════════════════

# MPI: 시스템 OpenMPI (/usr/bin/mpicxx). ⚠ Intel oneAPI 환경을 source 하지 말 것 —
# libmpi.so.12 가 섞여 링크가 깨진다 (MPM-STD_C 운용 경험).
# (CXX ?= 는 못 쓴다 — make 내장 기본값 CXX=g++ 가 이미 "정의됨"이라 무시된다.
#  기본값일 때만 mpicxx 로 바꾸고, 환경/명령행의 명시 지정은 존중한다.)
ifeq ($(origin CXX),default)
CXX := mpicxx
endif
MPIRUN ?= mpirun

# C++17 고정 — 이후 GPU 단계에서 쓸 nvc++ 23.7 이 지원하는 상한.
# OMPI_SKIP_MPICXX: OpenMPI 의 폐기된 C++ 바인딩 헤더를 끈다 (경고 소음 제거;
# 우리는 C API 만 쓴다).
CXXFLAGS ?= -std=c++17 -O2 -Wall -Wextra -DOMPI_SKIP_MPICXX

# FFTW3 — double 정밀도·직렬 (멀티스레드 금지 규칙). 압력 솔버(M8)가 쓴다.
# 정적 아카이브로 링크해서 실행 시 LD_LIBRARY_PATH 가 필요 없다.
# ⚠ ~/local/fftw3 (MPM-STD_C 용)은 nvhpc 로 빌드돼 g++ 링크가 깨진다
#   (__fd_sincos_1 미해결 + .so 는 nvhpc 쪽 OpenMPI 3.1.5 까지 끌고 와 시스템
#   OpenMPI 와 혼입). 그래서 같은 소스를 gcc 로 재빌드한 fftw3-gcc 를 쓴다.
#   GPU 단계는 cuFFT 라 nvhpc fftw 가 필요 없다.
FFTW3_DIR ?= /shared/home/wel1come1234/local/fftw3-gcc

# Makefile 이 참조하는 집계 변수 — 외부 라이브러리가 늘면 여기에 덧붙인다.
EXT_INC  := -I$(FFTW3_DIR)/include
EXT_LIBS := $(FFTW3_DIR)/lib/libfftw3.a -lm
DEPFLAGS := -MMD -MP

# ── GPU 빌드 (.cu 쌍둥이 TU; 2026-10-04 사용자 결정 — "실제 .cu 로, 그게 제일
#    빠르다") ────────────────────────────────────────────────────────────────
#   make channel GPU=1   (로그인 노드에서도 컴파일 가능 — cuda12.2 명시)
# * 모든 TU 를 mpic++ -cuda 로 컴파일한다 (.cpp 포함 — nvcc fat binary 를
#   섞으면 main 전 segfault 나는 함정의 회피책: 전부 한 컴파일러로)
# * 배열은 전부 디바이스 상주(cudaMalloc) — 통계 출력 때만 D2H (사용자 지시)
# * MPI = hpcx-2.15 (CUDA-aware; 시스템 UCX 1.14 가림 함정 — 실행 스크립트 참조)
# * FFT = cuFFT (-DMPMSTD_GPU_FFT), 분기 스위치는 -DMPMSTD_GPU 하나 (D5)
# * 산출물은 build_gpu/ 분리
ifdef GPU
# nvhpc 위치: 모듈이 NVHPC_ROOT 를 주면 그것 (이 클러스터 2026-10-17: `module load nvhpc/25.11_cuda12`
# → /apps/compiler/nvidia_hpc_sdk/25.11/Linux_x86_64/25.11, CUDA 12.9, hpc-x 2.25.1), 없으면 구 클러스터
# (gpu01/gpu02, 23.7) 의 절대 경로. 둘 다 명령행 NVHPC_DIR= 로 덮어쓸 수 있다.
NVHPC_DIR ?= $(or $(NVHPC_ROOT),/opt/nvidia/hpc-sdk/Linux_x86_64/23.7)
# CUDA 버전 = nvhpc 번들 (모듈의 KISTI_CUDA_VER=12.9.1 → 12.9; 구 클러스터 12.2).
CUDA_VER ?= $(if $(KISTI_CUDA_VER),$(shell echo $(KISTI_CUDA_VER) | cut -d. -f1,2),12.2)
# GPU 아키텍처: A100 cc80 (기본), H200 cc90, V100 cc70. 여러 개면 GPU_ARCH=cc80,cc90.
GPU_ARCH ?= cc80
# MPI: 모듈이 HPCX_MPI_DIR 를 주면 그것 (hpcx-2.25.1 은 ompi_hwloc/), 없으면 nvhpc 번들 hpcx-2.15 의 ompi/
# (Filtered_TDMA env_gpu_local.sh 로 검증된 조합 — 시스템 /lib 의 UCX 1.14 가 hpcx 를 가리는 함정이 있어,
# 실행 시 LD_LIBRARY_PATH 에 hpcx ucx/ompi 를 앞세워야 한다. 모듈 환경에서는 모듈이 한다.)
HPCX_DIR ?= $(NVHPC_DIR)/comm_libs/12.2/hpcx/hpcx-2.15
MPI_DIR  ?= $(or $(HPCX_MPI_DIR),$(HPCX_DIR)/ompi)
export PATH := $(NVHPC_DIR)/compilers/bin:$(PATH)
CXX := $(MPI_DIR)/bin/mpic++
MPIRUN := $(MPI_DIR)/bin/mpirun
CXXFLAGS := -std=c++17 -O2 -cuda -gpu=$(GPU_ARCH),cuda$(CUDA_VER) -DMPMSTD_GPU -DMPMSTD_GPU_FFT
EXT_INC := -I$(NVHPC_DIR)/math_libs/$(CUDA_VER)/targets/x86_64-linux/include
EXT_LIBS := -cudalib=cufft -lm
# nvc++ 는 -MP 를 모르고, .d 의 타깃 이름을 출력 basename 으로 쓰므로 -MT 로 전체
# 경로를 준다 (없으면 헤더 변경이 GPU 오브젝트를 재컴파일하지 못해 stale .o 로
# segfault/unknown key — 실측 2026-10-09, rbc_nob·dhvc). 재귀 확장(=) 이라 $@ 는 레시피 때 풀린다.
DEPFLAGS = -MMD -MT $@
BUILD := build_gpu
endif
