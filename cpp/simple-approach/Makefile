# Compiler
NVCC = nvcc

# Flags
NVCCFLAGS = -lgtest -lgtest_main

# Directories
SRC_DIR = src
BIN_DIR = bin

# Source files
SOURCES = $(wildcard $(SRC_DIR)/*.cu)

# bin/file1, bin/file2
# % is used to do pattern matching, see:
# https://www.gnu.org/software/make/manual/html_node/Pattern-Match.html
EXECUTABLES = $(patsubst $(SRC_DIR)/%.cu, $(BIN_DIR)/%, $(SOURCES))

# Rules
# executables will be the trigger for the next rule
all: create_dir $(EXECUTABLES)


# $@ bin/file_x
# $< src/file_x.cu
# https://www.gnu.org/software/make/manual/html_node/Automatic-Variables.html
$(BIN_DIR)/%: $(SRC_DIR)/%.cu
	$(NVCC) -o $@ $< $(NVCCFLAGS)

# The @ at the beginning prevents printing `mkdir -p bin` when calling make
# https://www.gnu.org/software/make/manual/html_node/Echoing.html
create_dir:
	@mkdir -p $(BIN_DIR)

print_executables:
	@$(info EXECUTABLES: $(EXECUTABLES))

.PHONY: all create_dir print_executables
