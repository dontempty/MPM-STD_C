include ../../Makefile.inc

BIN_DIR := ../../$(BUILD_DIR)/bin
LIB     := ../../$(BUILD_DIR)/lib/libmpmstd.a
TARGET  := $(BIN_DIR)/rbc

.PHONY: all clean

all: $(TARGET)

$(TARGET): main.cpp $(LIB)
	@mkdir -p $(BIN_DIR)
	@echo "  APP   $@"
	@$(CXX) $(ALL_CXXFLAGS) main.cpp $(LIB) $(EXT_LIBS) -o $@

clean:
	@rm -f $(TARGET)
