#include <stdio.h>
#include <cuda_runtime.h>

#define N 1048576

__device__ float d_A[N];
__device__ float d_B[N];
__device__ float d_C[N];

__global__ void vectorAdd() {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < N) {
        d_C[id] = d_A[id] + d_B[id];
    }
}

int main() {
    float h_A[N], h_B[N], h_C[N];

    for (int i = 0; i < N; i++) {
        h_A[i] = (float)i;
        h_B[i] = (float)i * 2.0f;
    }

    cudaMemcpyToSymbol(d_A, h_A, N * sizeof(float));
    cudaMemcpyToSymbol(d_B, h_B, N * sizeof(float));

    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>();
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    printf("Kernel execution time: %.4f ms\n", milliseconds);

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    double memClockHz = (double)prop.memoryClockRate * 1000.0;
    double busWidthBytes = (double)prop.memoryBusWidth / 8.0;
    double theoreticalBW = 2.0 * memClockHz * busWidthBytes / 1e9;
    printf("Theoretical bandwidth: %.2f GB/s\n", theoreticalBW);

    long long rBytes = (long long)N * 2 * sizeof(float);
    long long wBytes = (long long)N * sizeof(float);
    double totalBytes = (double)(rBytes + wBytes);
    double timeSeconds = milliseconds / 1000.0;
    double measuredBW = totalBytes / timeSeconds / 1e9;
    printf("Measured bandwidth:    %.2f GB/s\n", measuredBW);
    printf("Bandwidth efficiency:  %.1f%%\n", (measuredBW / theoreticalBW) * 100.0);

    cudaMemcpyFromSymbol(h_C, d_C, N * sizeof(float));

    int allCorrect = 1;
    for (int i = 0; i < N; i++) {
        float expected = h_A[i] + h_B[i];
        if (h_C[i] != expected) {
            allCorrect = 0;
            break;
        }
    }
    printf("Result correct: %s\n", allCorrect ? "YES" : "NO");

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
