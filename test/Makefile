include ../Makefile.inc

# ---------------------------------------------------------------
# Unit + integration tests.
# Each test is a single .cpp file that links against the library.
# ---------------------------------------------------------------

UNIT_SRCS := $(wildcard unit/*.cpp)
INT_SRCS  := $(wildcard integration/*.cpp)
TEST_SRCS := $(UNIT_SRCS) $(INT_SRCS)

OBJ_DIR := ../$(BUILD_DIR)/obj/test
BIN_DIR := ../$(BUILD_DIR)/bin/tests
LIB     := ../$(BUILD_DIR)/lib/libmpmstd.a

TEST_BINS := $(patsubst %.cpp,$(BIN_DIR)/%,$(TEST_SRCS))

.PHONY: all clean

all: $(TEST_BINS)

# Pattern rule: any test source -> matching binary
TEST_INCS := -I"$(ROOT_DIR)/test"

$(BIN_DIR)/%: %.cpp $(LIB)
	@mkdir -p $(dir $@) $(dir $(OBJ_DIR)/$*)
	@echo "  TEST  $@"
	@$(CXX) $(ALL_CXXFLAGS) $(TEST_INCS) $< $(LIB) $(EXT_LIBS) -o $@

clean:
	@rm -rf $(OBJ_DIR) $(BIN_DIR)
