# Makefile to compile the Ising model C code to both WASM and shared object (.so) files

# Set variables for paths and filenames
SOURCE = compdismatter/wasm/ising.c
WASM_OUTPUT = compdismatter/wasm/ising.wasm
SO_OUTPUT = compdismatter/wasm/ising.so
CFLAGS_WASM = -s SIDE_MODULE=2 -s EXPORTED_FUNCTIONS="['_mcmove']" -O3
CFLAGS_SO = -shared -fPIC -O3

# Default target (build both WASM and .so)
all: $(WASM_OUTPUT) $(SO_OUTPUT)

# Rule to compile the C source to WASM
$(WASM_OUTPUT): $(SOURCE)
	emcc $(SOURCE) $(CFLAGS_WASM) -o $(WASM_OUTPUT)

# Rule to compile the C source to a shared object (.so)
$(SO_OUTPUT): $(SOURCE)
	gcc $(SOURCE) $(CFLAGS_SO) -o $(SO_OUTPUT)

# Clean the build directory
clean:
	rm -f $(WASM_OUTPUT) $(SO_OUTPUT)

.PHONY: all clean
