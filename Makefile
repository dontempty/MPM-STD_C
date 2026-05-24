include Makefile.inc

.PHONY: all lib apps tests clean info

# Default: build library + tests (apps come online from M5)
all: lib tests

lib:
	@$(MAKE) -C src

tests: lib
	@$(MAKE) -C test

apps: lib
	@echo "(no apps yet — added at M5)"

clean:
	@rm -rf build/

# Diagnostic: print resolved settings.
info:
	@echo "BACKEND        = $(BACKEND)"
	@echo "PRECISION      = $(PRECISION)"
	@echo "CXX            = $(CXX)"
	@echo "CXXFLAGS       = $(CXXFLAGS)"
	@echo "CUFLAGS        = $(CUFLAGS)"
	@echo "DEFINES        = $(DEFINES)"
	@echo "INCS           = $(INCS)"
	@echo "EXT_LIBS       = $(EXT_LIBS)"
	@echo "BUILD_DIR      = $(BUILD_DIR)"
	@echo "PASCAL_TDMA_INC= $(PASCAL_TDMA_INC)"
	@echo "PASCAL_TDMA_LIB= $(PASCAL_TDMA_LIB)"
