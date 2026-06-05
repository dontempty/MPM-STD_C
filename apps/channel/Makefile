include ../../Makefile.inc

BIN_DIR := ../../$(BUILD_DIR)/bin
LIB     := ../../$(BUILD_DIR)/lib/libmpmstd.a
TARGET  := $(BIN_DIR)/channel
MAIN    := main.cpp

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(MAIN) $(LIB)
	@mkdir -p $(BIN_DIR)
	@echo "  APP   $@"
	@$(CXX) $(ALL_CXXFLAGS) $(MAIN) $(LIB) $(EXT_LIBS) -o $@

clean:
	@rm -f $(TARGET)
