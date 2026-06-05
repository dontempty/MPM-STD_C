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
# (BACKEND can be cpu or cuda; PRECISION can be double or single)

# -------- external dependencies --------
PASCAL_TDMA_DIR ?= /shared/home/wel1come1234/workspace/PaScaL_TDMA_C
PASCAL_TDMA_INC := $(PASCAL_TDMA_DIR)/build/include
PASCAL_TDMA_LIB := $(PASCAL_TDMA_DIR)/build/lib

FFTW3_DIR ?= /shared/home/wel1come1234/local/fftw3
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
  CXXFLAGS   := $(CXX_STD) $(COMMON_OPT) -fast
  CUFLAGS    := -cuda -gpu=cc80
  DEFINES    += -DMPMSTD_BACKEND_CUDA
  EXT_LIBS   := -L$(PASCAL_TDMA_LIB) -lpascal_tdma -cudalib=cufft -lm
  BUILD_DIR  := build/cuda

else
  $(error Unknown BACKEND='$(BACKEND)'. Use cpu or cuda)
endif

# -------- combined flags --------
ALL_CXXFLAGS := $(CXXFLAGS) $(WARN) $(DEFINES) $(INCS) $(DEPFLAGS)
ALL_CUFLAGS  := $(CXXFLAGS) $(CUFLAGS) $(DEFINES) $(INCS)

export BACKEND PRECISION CXX CXXFLAGS CUFLAGS ALL_CXXFLAGS ALL_CUFLAGS \
       BUILD_DIR DEFINES INCS EXT_LIBS WARN PASCAL_TDMA_INC PASCAL_TDMA_LIB
