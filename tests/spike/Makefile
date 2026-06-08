# =============================================================================
# tests/spike/Makefile — P-0.5 interface-spike build.
#
#   make cpu                 : system mpicxx, CPU-only (NO CUDA dependency). Runnable now.
#   make BACKEND=cuda gpu    : NVHPC mpic++ (module load nvhpc first), compiles the
#                              cpu/ host TUs AND the gpu/ .cu TUs into ONE binary that
#                              links the _gpu symbols + CUDA-aware MPI. Proves the
#                              dual-build model. (Run on a node with nvhpc; submit
#                              tests/spike/spike_gpu.sh)
#   make clean
#
# The spike compiles only the few existing host-single deps it needs
# (mpi_context / mpi_topology / subdomain) directly — no full libmpmstd build,
# no virtual Backend, no CUDA in the CPU path. (rev.2 §6)
# =============================================================================

ROOT := $(abspath ../..)
include $(ROOT)/Makefile.inc

# Makefile.inc INCS already has -I$(ROOT)/src; add the new tree root.
SPIKE_INC := -I"$(ROOT)/mpmstd"

# existing host-single deps (Backend-free, CUDA-free)
DEPS := \
  $(ROOT)/src/parallel/mpi/mpi_context.cpp \
  $(ROOT)/src/parallel/mpi/mpi_topology.cpp \
  $(ROOT)/src/parallel/mpi/subdomain.cpp

# new core/solve CPU implementations (cpu/ folder — always compiled)
CPU_IMPL := \
  $(ROOT)/mpmstd/core/cpu/halo_cpu.cpp \
  $(ROOT)/mpmstd/solve/cpu/banded_solver_cpu.cpp

BIN := $(ROOT)/$(BUILD_DIR)/spike

.PHONY: cpu gpu clean

# ---- CPU (single-shot link; proven) ----------------------------------------
cpu: $(BIN)/halo_poc $(BIN)/banded_poc
	@echo "  spike (cpu) -> $(BIN)"

$(BIN)/halo_poc: halo_poc.cpp $(CPU_IMPL) $(DEPS)
	@mkdir -p $(BIN)
	@echo "  CXX   $@"
	@$(CXX) $(ALL_CXXFLAGS) $(SPIKE_INC) $^ -o $@

$(BIN)/banded_poc: banded_poc.cpp $(CPU_IMPL) $(DEPS)
	@mkdir -p $(BIN)
	@echo "  CXX   $@"
	@$(CXX) $(ALL_CXXFLAGS) $(SPIKE_INC) $^ -o $@

# ---- GPU (ONE compiler: NVHPC mpic++) ---------------------------------------
# .cu compiled to objects first (the likely failure point, isolated), then host
# TUs + objects linked in one nvc++ invocation. Requires BACKEND=cuda.
gpu: $(BIN)/link_gpu
	@echo "  spike (gpu) -> $(BIN)"

$(BIN)/link_gpu: link_gpu.cpp $(CPU_IMPL) $(DEPS) \
                 $(ROOT)/mpmstd/core/gpu/halo_gpu.cu $(ROOT)/mpmstd/core/gpu/field_gpu.cu
	@if [ "$(BACKEND)" != "cuda" ]; then \
	  echo "ERROR: GPU spike needs the CUDA backend."; \
	  echo "       Run: make BACKEND=cuda gpu   (on a node with 'module load nvhpc')"; \
	  exit 1; fi
	@mkdir -p $(BIN)
	@echo "  NVCC  halo_gpu.o"
	@$(CXX) $(ALL_CUFLAGS) $(SPIKE_INC) -c $(ROOT)/mpmstd/core/gpu/halo_gpu.cu  -o $(BIN)/halo_gpu.o
	@echo "  NVCC  field_gpu.o"
	@$(CXX) $(ALL_CUFLAGS) $(SPIKE_INC) -c $(ROOT)/mpmstd/core/gpu/field_gpu.cu -o $(BIN)/field_gpu.o
	@echo "  LINK  $@"
	@$(CXX) $(ALL_CUFLAGS) $(SPIKE_INC) link_gpu.cpp $(CPU_IMPL) $(DEPS) \
	        $(BIN)/halo_gpu.o $(BIN)/field_gpu.o -o $@

clean:
	@rm -rf $(BIN)
