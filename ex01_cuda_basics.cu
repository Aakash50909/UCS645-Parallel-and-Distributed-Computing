#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

#define CUDA_CHECK(x) { cudaError_t e = x; if(e != cudaSuccess){ printf("CUDA error %s line %d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); exit(1); } }

__global__ void vectorScale(float *arr, float scalar, int n) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < n) {
        arr[id] = arr[id] * scalar;
    }
}

__global__ void squaredDiff(float *a, float *b, float *out, int n) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < n) {
        float diff = a[id] - b[id];
        out[id] = diff * diff;
    }
}

__global__ void vectorAdd(float *a, float *b, float *c, int n) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < n) {
        c[id] = a[id] + b[id];
    }
}

void runBandwidthBenchmark() {
    printf("\n--- Memory Bandwidth Benchmark ---\n");
    printf("%-15s %-15s %-15s\n", "Size (MB)", "H2D (MB/s)", "D2H (MB/s)");

    int sizes[] = {1, 8, 64, 256, 512};
    int numSizes = 5;

    for (int s = 0; s < numSizes; s++) {
        int mb = sizes[s];
        size_t bytes = (size_t)mb * 1024 * 1024;
        int count = bytes / sizeof(float);

        float *h_data = (float*)malloc(bytes);
        float *d_data;
        CUDA_CHECK(cudaMalloc((void**)&d_data, bytes));

        for (int i = 0; i < count; i++) h_data[i] = (float)i;

        cudaEvent_t t0, t1;
        cudaEventCreate(&t0);
        cudaEventCreate(&t1);

        cudaEventRecord(t0);
        CUDA_CHECK(cudaMemcpy(d_data, h_data, bytes, cudaMemcpyHostToDevice));
        cudaEventRecord(t1);
        cudaEventSynchronize(t1);
        float h2d_ms;
        cudaEventElapsedTime(&h2d_ms, t0, t1);

        cudaEventRecord(t0);
        CUDA_CHECK(cudaMemcpy(h_data, d_data, bytes, cudaMemcpyDeviceToHost));
        cudaEventRecord(t1);
        cudaEventSynchronize(t1);
        float d2h_ms;
        cudaEventElapsedTime(&d2h_ms, t0, t1);

        double h2d_bw = (double)bytes / (h2d_ms / 1000.0) / 1e6;
        double d2h_bw = (double)bytes / (d2h_ms / 1000.0) / 1e6;

        printf("%-15d %-15.1f %-15.1f\n", mb, h2d_bw, d2h_bw);

        free(h_data);
        cudaFree(d_data);
        cudaEventDestroy(t0);
        cudaEventDestroy(t1);
    }
}

void runSpeedupBenchmark() {
    printf("\n--- CPU vs GPU Speedup Benchmark ---\n");
    printf("%-10s %-15s %-15s %-15s %-10s\n", "N", "CPU (ms)", "GPU (ms)", "H2D (ms)", "Speedup");

    int powers[] = {10, 14, 18, 22, 26};
    int numPowers = 5;

    for (int p = 0; p < numPowers; p++) {
        int n = 1 << powers[p];
        size_t bytes = (size_t)n * sizeof(float);

        float *h_a = (float*)malloc(bytes);
        float *h_b = (float*)malloc(bytes);
        float *h_c = (float*)malloc(bytes);

        for (int i = 0; i < n; i++) {
            h_a[i] = (float)i * 0.1f;
            h_b[i] = (float)i * 0.2f;
        }

        struct timespec cpu_start, cpu_end;
        clock_gettime(CLOCK_MONOTONIC, &cpu_start);
        for (int i = 0; i < n; i++) h_c[i] = h_a[i] + h_b[i];
        clock_gettime(CLOCK_MONOTONIC, &cpu_end);
        double cpu_ms = ((double)(cpu_end.tv_sec - cpu_start.tv_sec) * 1000.0)
                      + ((double)(cpu_end.tv_nsec - cpu_start.tv_nsec) / 1e6);

        float *d_a, *d_b, *d_c;
        CUDA_CHECK(cudaMalloc((void**)&d_a, bytes));
        CUDA_CHECK(cudaMalloc((void**)&d_b, bytes));
        CUDA_CHECK(cudaMalloc((void**)&d_c, bytes));

        cudaEvent_t t0, t1;
        cudaEventCreate(&t0);
        cudaEventCreate(&t1);

        cudaEventRecord(t0);
        CUDA_CHECK(cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice));
        cudaEventRecord(t1);
        cudaEventSynchronize(t1);
        float h2d_ms;
        cudaEventElapsedTime(&h2d_ms, t0, t1);

        int threads = 256;
        int blocks = (n + threads - 1) / threads;

        cudaEventRecord(t0);
        vectorAdd<<<blocks, threads>>>(d_a, d_b, d_c, n);
        cudaEventRecord(t1);
        cudaEventSynchronize(t1);
        float gpu_ms;
        cudaEventElapsedTime(&gpu_ms, t0, t1);

        double speedup = cpu_ms / gpu_ms;
        printf("%-10d %-15.3f %-15.3f %-15.3f %-10.2f\n", n, cpu_ms, gpu_ms, h2d_ms, speedup);

        free(h_a); free(h_b); free(h_c);
        cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
        cudaEventDestroy(t0); cudaEventDestroy(t1);
    }
}

void runLaunchConfigAnalysis() {
    printf("\n--- Launch Configuration Analysis (N = 2^20) ---\n");
    int n = 1 << 20;
    size_t bytes = (size_t)n * sizeof(float);

    float *d_a, *d_b, *d_c;
    CUDA_CHECK(cudaMalloc((void**)&d_a, bytes));
    CUDA_CHECK(cudaMalloc((void**)&d_b, bytes));
    CUDA_CHECK(cudaMalloc((void**)&d_c, bytes));

    int blockSizes[] = {64, 128, 256, 512, 1024};
    int numSizes = 5;

    printf("%-20s %-15s %-15s\n", "Threads/Block", "Blocks", "Time (ms)");

    for (int s = 0; s < numSizes; s++) {
        int tpb = blockSizes[s];
        int blocks = (n + tpb - 1) / tpb;

        cudaEvent_t t0, t1;
        cudaEventCreate(&t0);
        cudaEventCreate(&t1);

        cudaEventRecord(t0);
        for (int iter = 0; iter < 10; iter++) {
            vectorAdd<<<blocks, tpb>>>(d_a, d_b, d_c, n);
        }
        cudaEventRecord(t1);
        cudaEventSynchronize(t1);
        float ms;
        cudaEventElapsedTime(&ms, t0, t1);

        printf("%-20d %-15d %-15.4f\n", tpb, blocks, ms / 10.0f);
        cudaEventDestroy(t0); cudaEventDestroy(t1);
    }

    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
}

__global__ void withDivergence(float *arr, int n) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < n) {
        if (threadIdx.x % 2 == 0) {
            arr[id] = arr[id] * 2.0f;
        } else {
            arr[id] = arr[id] + 1.0f;
        }
    }
}

__global__ void withoutDivergence(float *arr, int n) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < n) {
        arr[id] = arr[id] * 2.0f + (float)(id % 2);
    }
}

void runDivergenceExperiment() {
    printf("\n--- Warp Divergence Experiment ---\n");
    int n = 1 << 20;
    size_t bytes = (size_t)n * sizeof(float);

    float *d_arr;
    CUDA_CHECK(cudaMalloc((void**)&d_arr, bytes));

    int threads = 256;
    int blocks = (n + threads - 1) / threads;

    cudaEvent_t t0, t1;
    cudaEventCreate(&t0);
    cudaEventCreate(&t1);

    cudaEventRecord(t0);
    for (int i = 0; i < 100; i++) withDivergence<<<blocks, threads>>>(d_arr, n);
    cudaEventRecord(t1);
    cudaEventSynchronize(t1);
    float divergent_ms;
    cudaEventElapsedTime(&divergent_ms, t0, t1);

    cudaEventRecord(t0);
    for (int i = 0; i < 100; i++) withoutDivergence<<<blocks, threads>>>(d_arr, n);
    cudaEventRecord(t1);
    cudaEventSynchronize(t1);
    float nodiv_ms;
    cudaEventElapsedTime(&nodiv_ms, t0, t1);

    printf("With divergence:    %.3f ms\n", divergent_ms / 100.0f);
    printf("Without divergence: %.3f ms\n", nodiv_ms / 100.0f);
    printf("Slowdown from divergence: %.2fx\n", divergent_ms / nodiv_ms);

    cudaFree(d_arr);
    cudaEventDestroy(t0); cudaEventDestroy(t1);
}

int main() {
    printf("=== EX01: CUDA Basics ===\n");
    runBandwidthBenchmark();
    runSpeedupBenchmark();
    runLaunchConfigAnalysis();
    runDivergenceExperiment();
    return 0;
}
