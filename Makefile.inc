# ============================================================
#  MPM-STD (C++) build configuration
#
#  Usage:
#    make BACKEND=cpu  all      # CPU build (default)
#    make BACKEND=cuda all      # GPU build (NVHPC nvc++ -cuda)
#    make PRECISION=single all  # single precision (default: double)
# ============================================================

# -------- selectable knobs --------
BACKEND ?= cpu
PRECISION ?= double
# GPU compute capability for BACKEND=cuda: 80=A100 (Neuron gpu30-45),
# 90=H100/GH200 (gpu51-52 aarch64, gpu46-50/53-56 H200).
GPU_CC ?= 80
# (BACKEND can be cpu or cuda; PRECISION can be double or single)

# -------- external dependencies --------
# KISTI Neuron: PaScaL_TDMA_C lives in /scratch/x3319a05/TDMA/PaScaL_TDMA with
# two toolchain-matched builds — build_gcc (gcc/11.5.0 + openmpi-4.1.8, CPU
# only) and build_nv_sm80 (nvhpc/25.11_cuda12 hpcx + nvcc sm_80, CPU+CUDA).
PASCAL_TDMA_DIR ?= /scratch/x3319a05/TDMA/PaScaL_TDMA
ifeq ($(BACKEND),cuda)
  PASCAL_TDMA_BUILD ?= build_nv_sm$(GPU_CC)
else
  PASCAL_TDMA_BUILD ?= build_gcc
endif
PASCAL_TDMA_INC := $(PASCAL_TDMA_DIR)/$(PASCAL_TDMA_BUILD)/include
PASCAL_TDMA_LIB := $(PASCAL_TDMA_DIR)/$(PASCAL_TDMA_BUILD)/lib

FFTW3_DIR ?= /apps/libraries/fftw3/3.3.10/gcc/11.5.0
FFTW3_INC := $(FFTW3_DIR)/include
FFTW3_LIB := $(FFTW3_DIR)/lib

# -------- common flags --------
# OMPI_SKIP_MPICXX disables the deprecated OpenMPI C++ bindings header (we use
# the C interface only). Without it the build is flooded with "cast between
# incompatible function types" warnings emitted *inside* mpi.h itself.
WARN        := -Wall -Wextra -Wno-unused-parameter -Wno-cast-function-type
COMMON_DEFS := -DOMPI_SKIP_MPICXX
ROOT_DIR    := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
# Wrap paths in quotes because the project root path may contain shell-special
# characters such as `(` or `)` (e.g. .../MPM-STD(C++)/).
INCS        := -I"$(ROOT_DIR)/src" -I"$(PASCAL_TDMA_INC)" -I"$(FFTW3_INC)"
DEFINES     := $(COMMON_DEFS)
DEPFLAGS    := -MMD -MP
COMMON_OPT  := -O3
CXX_STD     := -std=c++17

# -------- precision --------
ifeq ($(PRECISION),single)
  DEFINES += -DMPMSTD_SINGLE_PRECISION
endif

# -------- backend-specific --------
ifeq ($(BACKEND),cpu)
  CXX        := mpicxx
  CXXFLAGS   := $(CXX_STD) $(COMMON_OPT) -march=native -fno-fast-math -fno-associative-math
  CUFLAGS    :=
  DEFINES    += -DMPMSTD_BACKEND_CPU
  EXT_LIBS   := -L$(PASCAL_TDMA_LIB) -lpascal_tdma -L$(FFTW3_LIB) -lfftw3 -lm
  BUILD_DIR  := build/cpu

else ifeq ($(BACKEND),cuda)
  # NVHPC's mpic++ understands -cuda for .cu files; one compiler does both.
  CXX        := mpic++
  # Host ISA target. nvc++ defaults to the BUILD host CPU — Neuron login nodes
  # are Intel Xeon (AVX-512) while every x86 GPU node is AMD EPYC (no AVX-512),
  # which SIGILLs at runtime (dhvc write_statistics, job 862652). haswell =
  # AVX2/FMA common denominator of all Neuron x86 nodes. Pass HOST_TP= (empty)
  # on aarch64 (GH200) where -tp is not accepted.
  HOST_TP    ?= haswell
  CXXFLAGS   := $(CXX_STD) $(COMMON_OPT) -fast $(if $(HOST_TP),-tp=$(HOST_TP))
  CUFLAGS    := -cuda -gpu=cc$(GPU_CC)
  DEFINES    += -DMPMSTD_BACKEND_CUDA
  # Host-only apps link WITHOUT -cuda, so nvc++ turns -cudalib=cufft into a
  # bare -lcufft without adding the NVHPC math_libs search path — supply it
  # explicitly (KISTI Neuron, nvhpc/25.11_cuda12).
  NVHPC_MATH_LIB ?= /apps/compiler/nvidia_hpc_sdk/25.11/Linux_x86_64/25.11/math_libs/12.9/lib64
  EXT_LIBS   := -L$(PASCAL_TDMA_LIB) -lpascal_tdma -L$(NVHPC_MATH_LIB) -cudalib=cufft -lm
  # Some SDK layouts (e.g. ARM nvhpc 24.5) don't put cufft.h on the default
  # include path — pass NVHPC_MATH_INC=<sdk>/math_libs/include to add it.
  NVHPC_MATH_INC ?=
  ifneq ($(NVHPC_MATH_INC),)
    INCS += -I"$(NVHPC_MATH_INC)"
  endif
  BUILD_DIR  := build/cuda

else
  $(error Unknown BACKEND='$(BACKEND)'. Use cpu or cuda)
endif

# -------- combined flags --------
ALL_CXXFLAGS := $(CXXFLAGS) $(WARN) $(DEFINES) $(INCS) $(DEPFLAGS)
ALL_CUFLAGS  := $(CXXFLAGS) $(CUFLAGS) $(DEFINES) $(INCS)

export BACKEND PRECISION CXX CXXFLAGS CUFLAGS ALL_CXXFLAGS ALL_CUFLAGS \
       BUILD_DIR DEFINES INCS EXT_LIBS WARN PASCAL_TDMA_INC PASCAL_TDMA_LIB
