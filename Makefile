NVCC = nvcc
ARCH = -arch=sm_86

all: partA partB partC

partA: partA_device_query.cu
	$(NVCC) $(ARCH) -o partA partA_device_query.cu

partB: partB_array_sum.cu
	$(NVCC) $(ARCH) -o partB partB_array_sum.cu

partC: partC_matrix_add.cu
	$(NVCC) $(ARCH) -o partC partC_matrix_add.cu

clean:
	rm -f partA partB partC
