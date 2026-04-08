#include <stdio.h>
#include <cuda_runtime.h>

int main() {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    printf("Name: %s\n", prop.name);
    printf("Compute Capability: %d.%d\n", prop.major, prop.minor);
    printf("Max Threads Per Block: %d\n", prop.maxThreadsPerBlock);
    printf("Max Block Dim: %d x %d x %d\n", prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
    printf("Max Grid Dim: %d x %d x %d\n", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
    printf("Warp Size: %d\n", prop.warpSize);
    printf("Total Global Memory: %.2f MB\n", prop.totalGlobalMem / (1024.0 * 1024.0));
    printf("Shared Memory Per Block: %zu bytes\n", prop.sharedMemPerBlock);
    printf("Constant Memory: %zu bytes\n", prop.totalConstMem);
    printf("Multiprocessors: %d\n", prop.multiProcessorCount);
    printf("Double Precision Support: %s\n", prop.major >= 2 ? "Yes" : "No");

    return 0;
}
