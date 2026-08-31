# =============================================================================
# Makefile — MPM-STD build (rev.2). Normal layout: library in src/, apps in
# apps/, build in build/$(BACKEND)/.
#
# Builds libmpmstd from src/ (the 6 numeric layers core solve equation physics
# post driver + the reused host infrastructure common/ parallel/ config/ grid/
# boundary/ linear_solver/) + apps + tests into build/$(BACKEND)/.
#
#   make cpu                 : system mpicxx, CUDA-free (runs on the login node)
#   make BACKEND=cuda gpu     : NVHPC mpic++ (module load nvhpc; submit on gpu0x)
#   make clean
#
# Dual-build model: backend is selected by FOLDER — every layer has cpu/ (host)
# and gpu/ (.cu). `make cpu` compiles cpu/ + backend-agnostic only; `make gpu`
# (BACKEND=cuda) additionally compiles gpu/ .cu via the SAME NVHPC compiler. The
# function-name _cpu/_gpu suffix is the caller's selector.
# =============================================================================

include Makefile.inc        # CXX, ALL_CXXFLAGS, ALL_CUFLAGS, BACKEND, EXT_LIBS, ROOT_DIR, INCS

INC    := -I"$(ROOT_DIR)/src"     # library lives under src/ (core/, equation/, … + infra); -I src is also in INCS

# The pressure engine uses host FFTW even in the GPU build, but Makefile.inc's
# cuda EXT_LIBS omits -lfftw3 (it only has cufft). Add it so a cuda-compiled
# channel (nvc++ host code + FFTW Poisson) links on the compute nodes.
ifeq ($(BACKEND),cuda)
  EXT_LIBS += -L$(FFTW3_LIB) -lfftw3
endif
# BUILD_TAG lets one BACKEND coexist in several output trees (e.g. x86 cc80
# in build/cuda vs GH200 aarch64 cc90 in build/cuda_gh200).
BUILD_TAG ?= $(BACKEND)
BUILD  := $(ROOT_DIR)/build/$(BUILD_TAG)
OBJ    := $(BUILD)/obj
LIB    := $(BUILD)/lib/libmpmstd.a
BINDIR := $(BUILD)/bin

LAYERS := src/core src/solve src/equation src/physics src/post src/driver

# ---- library sources (folder-based dual build) -----------------------------
LIB_CPU_CPP := $(shell find $(LAYERS) -path '*/cpu/*' -name '*.cpp' 2>/dev/null)
LIB_GPU_CU  := $(shell find $(LAYERS) -path '*/gpu/*' -name '*.cu'  2>/dev/null)
LIB_AGN_CPP := $(shell find $(LAYERS) -name '*.cpp' -not -path '*/cpu/*' -not -path '*/gpu/*' 2>/dev/null)

# existing host-single infra reused verbatim (rev.2: host metadata not duplicated).
# (BC ghost/tdma-row + the numeric kernels were PORTED into mpmstd; these remaining
#  pieces are pure host-single config/grid/topology/BC-data/TDMA-backend utilities
#  with no ScalarField/Backend dependency.)
SRC_INFRA := \
  parallel/mpi/mpi_context.cpp \
  parallel/mpi/mpi_topology.cpp \
  parallel/mpi/subdomain.cpp \
  config/config.cpp \
  config/logger.cpp \
  grid/grid.cpp \
  grid/stretching.cpp \
  boundary/problem.cpp \
  boundary/problem_loader.cpp \
  boundary/domain_topology.cpp \
  boundary/face_bc.cpp \
  linear_solver/tdma/tdma_registry.cpp \
  linear_solver/tdma/pascal_tdma_cpu_backend.cpp

ifeq ($(BACKEND),cuda)
  LIB_CU     := $(LIB_GPU_CU)
  # rev.2 dual model: cpu/ .cpp are the CPU backend (compiled by nvc++ as PLAIN
  # HOST C++ — NO -cuda, -DMPMSTD_BACKEND_CPU), gpu/ .cu are the CUDA backend
  # (-cuda, -DMPMSTD_BACKEND_CUDA). Compiling host .cpp WITH -cuda made nvc++ ICE
  # (signal 6, lower_dynamic_init) on dynamic-init code (e.g. boundary/problem.cpp
  # std::function tables); and -DMPMSTD_BACKEND_CPU stops macros.hpp pulling
  # <cuda_runtime.h>, so no -cuda is needed for host TUs. Strip the GCC-only
  # WARN/-MP that nvc++ rejects. (Shared Makefile.inc untouched → CPU baseline safe.)
  HOST_FLAGS := $(CXXFLAGS) -DOMPI_SKIP_MPICXX -DMPMSTD_BACKEND_CPU $(INCS) $(INC)
  # CUDA-aware host flags: -cuda enables cuda_runtime.h for wrappers that call
  # CUDA runtime APIs (e.g. pascal_tdma_cuda_backend). No dynamic-init std::function
  # in these files so the nvc++ ICE that affects boundary/problem.cpp is not triggered.
  CUDA_HOST_FLAGS := $(CXXFLAGS) -cuda -DOMPI_SKIP_MPICXX -DMPMSTD_BACKEND_CUDA $(INCS) $(INC)
  DEV_FLAGS  := $(ALL_CUFLAGS) $(INC)
else
  LIB_CU     :=
  HOST_FLAGS := $(ALL_CXXFLAGS) $(INC)
  CUDA_HOST_FLAGS :=
  DEV_FLAGS  := $(ALL_CUFLAGS) $(INC)
endif
LIB_CPP := $(LIB_CPU_CPP) $(LIB_AGN_CPP)

LIB_OBJS := $(patsubst %.cpp,$(OBJ)/%.o,$(LIB_CPP)) \
            $(patsubst %.cu,$(OBJ)/%.o,$(LIB_CU)) \
            $(patsubst %.cpp,$(OBJ)/_src/%.o,$(SRC_INFRA))

ifeq ($(BACKEND),cuda)
  # pascal_tdma_cuda_backend.cpp needs MPMSTD_BACKEND_CUDA (not _CPU) to activate
  # the CUDA runtime calls. Compiled as plain host C++ (no -cuda flag).
  CUDA_WRAP_OBJ := $(OBJ)/_src/linear_solver/tdma/pascal_tdma_cuda_backend.o
  LIB_OBJS += $(CUDA_WRAP_OBJ)
$(CUDA_WRAP_OBJ): $(ROOT_DIR)/src/linear_solver/tdma/pascal_tdma_cuda_backend.cpp
	@mkdir -p $(dir $@)
	@echo "  CXX   src/linear_solver/tdma/pascal_tdma_cuda_backend.cpp"
	@$(CXX) $(CUDA_HOST_FLAGS) -c $< -o $@

  # tdma_registry.cpp selects CPU vs CUDA backend via #ifdef MPMSTD_BACKEND_CUDA.
  # The generic SRC_INFRA rule uses HOST_FLAGS (no CUDA macro), which would create
  # PaScaLTDMACpuBackend even in the CUDA build. Override with the CUDA macro so
  # the registry creates PaScaLTDMACudaBackend — required for scalar_gpu.cu which
  # does static_cast<PaScaLTDMACudaBackend&>(tdma.get(d)). No -cuda flag needed
  # here since this file does not call CUDA runtime APIs directly.
  TDMA_REG_OBJ := $(OBJ)/_src/linear_solver/tdma/tdma_registry.o
$(TDMA_REG_OBJ): $(ROOT_DIR)/src/linear_solver/tdma/tdma_registry.cpp
	@mkdir -p $(dir $@)
	@echo "  CXX   src/linear_solver/tdma/tdma_registry.cpp"
	@$(CXX) $(HOST_FLAGS) -DMPMSTD_BACKEND_CUDA -c $< -o $@
endif

# ---- apps + tests -----------------------------------------------------------
APPS      := channel cavity dhvc rbc validate rbc_cpu_ob rbc_cpu_nob
ifeq ($(BACKEND),cuda)
  APPS    += rbc_gpu_ob rbc_gpu_nob dhvc_gpu
endif
APP_BINS  := $(patsubst %,$(BINDIR)/apps/%,$(APPS))
TEST_CPP  := $(shell find $(ROOT_DIR)/tests/unit $(ROOT_DIR)/tests/integration $(ROOT_DIR)/tests/regression -name '*.cpp' 2>/dev/null)
TEST_BINS := $(patsubst $(ROOT_DIR)/tests/%.cpp,$(BINDIR)/tests/%,$(TEST_CPP))

# =============================================================================
.PHONY: all cpu gpu lib apps tests clean info
cpu: all
gpu: all
all: lib apps tests

lib: $(LIB)
	@echo "  LIB   $(LIB)"

$(LIB): $(LIB_OBJS)
	@mkdir -p $(dir $@)
	@ar rcs $@ $^

# compile rules
$(OBJ)/%.o: %.cpp
	@mkdir -p $(dir $@)
	@echo "  CXX   $<"
	@$(CXX) $(HOST_FLAGS) -c $< -o $@
$(OBJ)/%.o: %.cu
	@mkdir -p $(dir $@)
	@echo "  NVCC  $<"
	@$(CXX) $(DEV_FLAGS) -c $< -o $@

# explicit .cuh header dependencies — add when a .cu includes a heavy header
$(OBJ)/src/equation/pressure/gpu/pressure_gpu.o: \
    src/equation/pressure/gpu/pressure_engine_gpu.cuh
$(OBJ)/_src/%.o: $(ROOT_DIR)/src/%.cpp
	@mkdir -p $(dir $@)
	@echo "  CXX   src/$*.cpp"
	@$(CXX) $(HOST_FLAGS) -c $< -o $@

apps: lib $(APP_BINS)
$(BINDIR)/apps/%: apps/%/main.cpp $(LIB)
	@mkdir -p $(dir $@) $(OBJ)/apps
	@echo "  APP   $@"
	@$(CXX) $(HOST_FLAGS) -c $< -o $(OBJ)/apps/$*.o
	@$(CXX) $(HOST_FLAGS) $(OBJ)/apps/$*.o $(LIB) $(EXT_LIBS) -o $@

# RBC split variants
$(BINDIR)/apps/rbc_cpu_ob: apps/rbc/main_cpu_ob.cpp $(LIB)
	@mkdir -p $(dir $@) $(OBJ)/apps
	@echo "  APP   $@"
	@$(CXX) $(HOST_FLAGS) -c $< -o $(OBJ)/apps/rbc_cpu_ob.o
	@$(CXX) $(HOST_FLAGS) $(OBJ)/apps/rbc_cpu_ob.o $(LIB) $(EXT_LIBS) -o $@

$(BINDIR)/apps/rbc_cpu_nob: apps/rbc/main_cpu_nob.cpp $(LIB)
	@mkdir -p $(dir $@) $(OBJ)/apps
	@echo "  APP   $@"
	@$(CXX) $(HOST_FLAGS) -c $< -o $(OBJ)/apps/rbc_cpu_nob.o
	@$(CXX) $(HOST_FLAGS) $(OBJ)/apps/rbc_cpu_nob.o $(LIB) $(EXT_LIBS) -o $@

$(BINDIR)/apps/rbc_gpu_ob: apps/rbc/main_gpu_ob.cu $(LIB)
	@mkdir -p $(dir $@) $(OBJ)/apps
	@echo "  APP   $@"
	@$(CXX) $(DEV_FLAGS) -c $< -o $(OBJ)/apps/rbc_gpu_ob.o
	@$(CXX) $(DEV_FLAGS) $(OBJ)/apps/rbc_gpu_ob.o $(LIB) $(EXT_LIBS) -o $@

$(BINDIR)/apps/rbc_gpu_nob: apps/rbc/main_gpu_nob.cu $(LIB)
	@mkdir -p $(dir $@) $(OBJ)/apps
	@echo "  APP   $@"
	@$(CXX) $(DEV_FLAGS) -c $< -o $(OBJ)/apps/rbc_gpu_nob.o
	@$(CXX) $(DEV_FLAGS) $(OBJ)/apps/rbc_gpu_nob.o $(LIB) $(EXT_LIBS) -o $@

$(BINDIR)/apps/dhvc_gpu: apps/dhvc/main_gpu.cu $(LIB)
	@mkdir -p $(dir $@) $(OBJ)/apps
	@echo "  APP   $@"
	@$(CXX) $(DEV_FLAGS) -c $< -o $(OBJ)/apps/dhvc_gpu.o
	@$(CXX) $(DEV_FLAGS) $(OBJ)/apps/dhvc_gpu.o $(LIB) $(EXT_LIBS) -o $@

tests: lib $(TEST_BINS)
$(BINDIR)/tests/%: $(ROOT_DIR)/tests/%.cpp $(LIB)
	@mkdir -p $(dir $@)
	@echo "  TEST  $@"
	@$(CXX) $(HOST_FLAGS) -I"$(ROOT_DIR)/tests" $< $(LIB) $(EXT_LIBS) -o $@

-include $(LIB_OBJS:.o=.d)

clean:
	@rm -rf $(BUILD)

info:
	@echo "BACKEND   = $(BACKEND)"
	@echo "LIB       = $(LIB)"
	@echo "LIB_CPP   = $(LIB_CPP)"
	@echo "LIB_CU    = $(LIB_CU)"
	@echo "APP_BINS  = $(APP_BINS)"
	@echo "TEST_BINS = $(TEST_BINS)"
