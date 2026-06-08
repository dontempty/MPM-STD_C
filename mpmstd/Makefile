# =============================================================================
# mpmstd/Makefile — refactored MPM-STD build (rev.2, 병행 신축).
#
# Builds libmpmstd (6 layers: core solve equation physics post driver) + apps +
# tests into build/$(BACKEND)/mpmstd/. The old src/ tree is UNTOUCHED and remains
# the regression baseline (it builds into build/$(BACKEND)/ — no collision).
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

include ../Makefile.inc        # CXX, ALL_CXXFLAGS, ALL_CUFLAGS, BACKEND, EXT_LIBS, ROOT_DIR, INCS

INC    := -I"$(ROOT_DIR)/mpmstd"     # so #include "core/..." resolves (-I src is already in INCS)
BUILD  := $(ROOT_DIR)/build/$(BACKEND)/mpmstd
OBJ    := $(BUILD)/obj
LIB    := $(BUILD)/lib/libmpmstd.a
BINDIR := $(BUILD)/bin

LAYERS := core solve equation physics post driver

# ---- library sources (folder-based dual build) -----------------------------
LIB_CPU_CPP := $(shell find $(LAYERS) -path '*/cpu/*' -name '*.cpp' 2>/dev/null)
LIB_GPU_CU  := $(shell find $(LAYERS) -path '*/gpu/*' -name '*.cu'  2>/dev/null)
LIB_AGN_CPP := $(shell find $(LAYERS) -name '*.cpp' -not -path '*/cpu/*' -not -path '*/gpu/*' 2>/dev/null)

# existing host-single infra reused verbatim (rev.2: host metadata not duplicated)
SRC_INFRA := \
  parallel/mpi/mpi_context.cpp \
  parallel/mpi/mpi_topology.cpp \
  parallel/mpi/subdomain.cpp

ifeq ($(BACKEND),cuda)
  LIB_CU     := $(LIB_GPU_CU)
  # In the GPU build EVERY TU goes through nvc++ with -cuda (= ALL_CUFLAGS): host
  # cpu/ .cpp too. Two reasons: (1) nvc++ rejects GCC-only WARN/-MP that live in
  # ALL_CXXFLAGS; (2) macros.hpp pulls <cuda_runtime.h> under -DMPMSTD_BACKEND_CUDA,
  # whose include path nvc++ only adds with -cuda. Compiling pure-host .cpp with
  # -cuda is harmless (no device code) and is exactly what the P-0.5 spike proved.
  # (Shared Makefile.inc is untouched, so the validated CPU baseline is unaffected.)
  HOST_FLAGS := $(ALL_CUFLAGS) $(INC)
  DEV_FLAGS  := $(ALL_CUFLAGS) $(INC)
else
  LIB_CU     :=
  HOST_FLAGS := $(ALL_CXXFLAGS) $(INC)
  DEV_FLAGS  := $(ALL_CUFLAGS) $(INC)
endif
LIB_CPP := $(LIB_CPU_CPP) $(LIB_AGN_CPP)

LIB_OBJS := $(patsubst %.cpp,$(OBJ)/%.o,$(LIB_CPP)) \
            $(patsubst %.cu,$(OBJ)/%.o,$(LIB_CU)) \
            $(patsubst %.cpp,$(OBJ)/_src/%.o,$(SRC_INFRA))

# ---- apps + tests -----------------------------------------------------------
APPS      := channel cavity dhvc rbc
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
$(OBJ)/_src/%.o: $(ROOT_DIR)/src/%.cpp
	@mkdir -p $(dir $@)
	@echo "  CXX   src/$*.cpp"
	@$(CXX) $(HOST_FLAGS) -c $< -o $@

apps: lib $(APP_BINS)
$(BINDIR)/apps/%: apps/%/main.cpp $(LIB)
	@mkdir -p $(dir $@)
	@echo "  APP   $@"
	@$(CXX) $(HOST_FLAGS) $< $(LIB) $(EXT_LIBS) -o $@

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
