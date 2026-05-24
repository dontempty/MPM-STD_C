include ../Makefile.inc

# ---------------------------------------------------------------
# Collect sources.
#
# Convention:
#   - Files named *_cpu.cpp  → compiled only when BACKEND=cpu
#   - Files named *_cuda.cpp → compiled only when BACKEND=cuda
#   - *.cu files             → compiled only when BACKEND=cuda
#   - All other *.cpp files  → compiled in both builds
# ---------------------------------------------------------------

ALL_CPP := $(shell find . -name "*.cpp")
ALL_CU  := $(shell find . -name "*.cu")

ifeq ($(BACKEND),cpu)
  SRCS_CPP := $(filter-out %_cuda.cpp,$(ALL_CPP))
  SRCS_CU  :=
else
  SRCS_CPP := $(filter-out %_cpu.cpp,$(ALL_CPP))
  SRCS_CU  := $(ALL_CU)
endif

OBJ_DIR := ../$(BUILD_DIR)/obj
LIB_DIR := ../$(BUILD_DIR)/lib

OBJS_CPP := $(patsubst ./%.cpp,$(OBJ_DIR)/%.o,$(SRCS_CPP))
OBJS_CU  := $(patsubst ./%.cu,$(OBJ_DIR)/%.o,$(SRCS_CU))
OBJS     := $(OBJS_CPP) $(OBJS_CU)

LIB := $(LIB_DIR)/libmpmstd.a

.PHONY: all clean

all: $(LIB)

$(LIB): $(OBJS)
	@mkdir -p $(dir $@)
	@echo "  AR    $@"
	@ar rcs $@ $^

# Compile .cpp sources.
$(OBJ_DIR)/%.o: %.cpp
	@mkdir -p $(dir $@)
	@echo "  CXX   $<"
	@$(CXX) $(ALL_CXXFLAGS) -c $< -o $@

# Compile .cu sources (CUDA build only — pattern unused when SRCS_CU empty).
$(OBJ_DIR)/%.o: %.cu
	@mkdir -p $(dir $@)
	@echo "  NVCC  $<"
	@$(CXX) $(ALL_CUFLAGS) -c $< -o $@

# Auto-generated dependency tracking.
-include $(OBJS:.o=.d)

clean:
	@rm -rf $(OBJ_DIR) $(LIB)
