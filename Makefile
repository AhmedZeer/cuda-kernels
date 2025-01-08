# Compiler and flags
NVCC = nvcc
CFLAGS = -std=c++11 -Xcompiler -Wall

# Project structure
SRC_DIR = src
GEMM_HEADERS = $(SRC_DIR)/GEMM/headers
GEMM_KERNELS = $(SRC_DIR)/GEMM/kernels
GEMM_RUNNERS = $(SRC_DIR)/GEMM/runners
UTILS = $(SRC_DIR)/utils

# Output binary
OUTPUT = benchmark_gemm

# Source files
SOURCES = $(SRC_DIR)/benchmark_gemm.cu \
          $(GEMM_RUNNERS)/naive_runner.cu \
          $(GEMM_KERNELS)/naive.cu \
          $(UTILS)/util.cu

# Object files
OBJECTS = $(SOURCES:.cu=.o)

# Include paths
INCLUDES = -I$(GEMM_HEADERS) -I$(UTILS)

# Default target
all: $(OUTPUT)

# Build the executable
$(OUTPUT): $(OBJECTS)
	$(NVCC) $(CFLAGS) $(OBJECTS) -o $@

# Compile CUDA and C source files into object files
%.o: %.cu
	$(NVCC) $(CFLAGS) $(INCLUDES) -c $< -o $@

# Clean up build artifacts
clean:
	rm -f $(OBJECTS) $(OUTPUT)

# Phony targets
.PHONY: all clean
