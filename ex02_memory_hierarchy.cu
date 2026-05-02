#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

#define CUDA_CHECK(x) { cudaError_t e = x; if(e != cudaSuccess){ printf("CUDA error %s line %d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); exit(1); } }

#define BLOCK_SIZE 256

__global__ void naiveSum(float *arr, float *result, int n) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        float total = 0.0f;
        for (int i = 0; i < n; i++) {
            total = total + arr[i];
        }
        result[0] = total;
    }
}

__global__ void sharedMemSum(float *arr, float *blockSums, int n) {
    __shared__ float tile[BLOCK_SIZE];
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    if (id < n) {
        tile[tid] = arr[id];
    } else {
        tile[tid] = 0.0f;
    }
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride = stride / 2) {
        if (tid < stride) {
            tile[tid] = tile[tid] + tile[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        blockSums[blockIdx.x] = tile[0];
    }
}

__global__ void sharedMemMax(float *arr, float *blockMaxes, int n) {
    __shared__ float tile[BLOCK_SIZE];
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    if (id < n) {
        tile[tid] = arr[id];
    } else {
        tile[tid] = -1e30f;
    }
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride = stride / 2) {
        if (tid < stride) {
            if (tile[tid + stride] > tile[tid]) {
                tile[tid] = tile[tid + stride];
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        blockMaxes[blockIdx.x] = tile[0];
    }
}

__global__ void warpShuffleSum(float *arr, float *blockSums, int n) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    float val = (id < n) ? arr[id] : 0.0f;

    for (int offset = 16; offset > 0; offset = offset / 2) {
        val = val + __shfl_down_sync(0xffffffff, val, offset);
    }

    if (threadIdx.x % 32 == 0) {
        atomicAdd(&blockSums[blockIdx.x], val);
    }
}

void runReductionBenchmark() {
    printf("\n--- Reduction Strategies Benchmark (N = 2^20) ---\n");
    int n = 1 << 20;
    size_t bytes = (size_t)n * sizeof(float);

    float *h_arr = (float*)malloc(bytes);
    double cpu_ref = 0.0;
    for (int i = 0; i < n; i++) {
        h_arr[i] = 1.0f;
        cpu_ref = cpu_ref + h_arr[i];
    }

    float *d_arr, *d_result;
    CUDA_CHECK(cudaMalloc((void**)&d_arr, bytes));
    CUDA_CHECK(cudaMalloc((void**)&d_result, sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_arr, h_arr, bytes, cudaMemcpyHostToDevice));

    cudaEvent_t t0, t1;
    cudaEventCreate(&t0);
    cudaEventCreate(&t1);

    float h_result;

    cudaMemset(d_result, 0, sizeof(float));
    cudaEventRecord(t0);
    naiveSum<<<1, 1>>>(d_arr, d_result, n);
    cudaEventRecord(t1);
    cudaEventSynchronize(t1);
    float naive_ms;
    cudaEventElapsedTime(&naive_ms, t0, t1);
    cudaMemcpy(&h_result, d_result, sizeof(float), cudaMemcpyDeviceToHost);
    printf("Naive sum:         %.0f  time: %.1f us  correct: %s\n",
           h_result, naive_ms * 1000, fabsf(h_result - cpu_ref) < 0.1f ? "YES" : "NO");

    int blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    float *d_blockSums;
    CUDA_CHECK(cudaMalloc((void**)&d_blockSums, blocks * sizeof(float)));
    float *h_blockSums = (float*)malloc(blocks * sizeof(float));

    cudaEventRecord(t0);
    sharedMemSum<<<blocks, BLOCK_SIZE>>>(d_arr, d_blockSums, n);
    cudaEventRecord(t1);
    cudaEventSynchronize(t1);
    float shared_ms;
    cudaEventElapsedTime(&shared_ms, t0, t1);
    cudaMemcpy(h_blockSums, d_blockSums, blocks * sizeof(float), cudaMemcpyDeviceToHost);
    float shared_total = 0;
    for (int i = 0; i < blocks; i++) shared_total = shared_total + h_blockSums[i];
    double shared_bw = (double)bytes / (shared_ms / 1000.0) / 1e9;
    printf("Shared mem sum:    %.0f  time: %.1f us  BW: %.1f GB/s  correct: %s\n",
           shared_total, shared_ms * 1000, shared_bw, fabsf(shared_total - cpu_ref) < 0.1f ? "YES" : "NO");

    cudaMemset(d_blockSums, 0, blocks * sizeof(float));
    cudaEventRecord(t0);
    warpShuffleSum<<<blocks, BLOCK_SIZE>>>(d_arr, d_blockSums, n);
    cudaEventRecord(t1);
    cudaEventSynchronize(t1);
    float warp_ms;
    cudaEventElapsedTime(&warp_ms, t0, t1);
    cudaMemcpy(h_blockSums, d_blockSums, blocks * sizeof(float), cudaMemcpyDeviceToHost);
    float warp_total = 0;
    for (int i = 0; i < blocks; i++) warp_total = warp_total + h_blockSums[i];
    double warp_bw = (double)bytes / (warp_ms / 1000.0) / 1e9;
    printf("Warp shuffle sum:  %.0f  time: %.1f us  BW: %.1f GB/s  correct: %s\n",
           warp_total, warp_ms * 1000, warp_bw, fabsf(warp_total - cpu_ref) < 0.1f ? "YES" : "NO");

    free(h_arr); free(h_blockSums);
    cudaFree(d_arr); cudaFree(d_result); cudaFree(d_blockSums);
    cudaEventDestroy(t0); cudaEventDestroy(t1);
}

__global__ void bankConflictKernel(float *data, int stride, int n) {
    __shared__ float tile[1024];
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;
    int idx = (tid * stride) % blockDim.x;
    if (id < n) tile[idx] = data[id];
    __syncthreads();
    if (id < n) data[id] = tile[idx] * 2.0f;
}

void runBankConflictBenchmark() {
    printf("\n--- Bank Conflict Profiling ---\n");
    printf("%-10s %-15s\n", "Stride", "Time (us)");

    int n = 1 << 20;
    size_t bytes = (size_t)n * sizeof(float);
    float *d_data;
    CUDA_CHECK(cudaMalloc((void**)&d_data, bytes));

    int strides[] = {1, 2, 4, 8, 16, 32};
    int numStrides = 6;

    cudaEvent_t t0, t1;
    cudaEventCreate(&t0);
    cudaEventCreate(&t1);

    for (int s = 0; s < numStrides; s++) {
        int stride = strides[s];
        int blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;

        cudaEventRecord(t0);
        for (int iter = 0; iter < 100; iter++) {
            bankConflictKernel<<<blocks, BLOCK_SIZE>>>(d_data, stride, n);
        }
        cudaEventRecord(t1);
        cudaEventSynchronize(t1);
        float ms;
        cudaEventElapsedTime(&ms, t0, t1);

        printf("%-10d %-15.2f\n", stride, ms / 100.0f * 1000.0f);
    }

    cudaFree(d_data);
    cudaEventDestroy(t0); cudaEventDestroy(t1);
}

__global__ void paddedSharedMem(float *data, int n) {
    __shared__ float tile[16][17];
    int row = threadIdx.y;
    int col = threadIdx.x;
    int id = (blockIdx.x * blockDim.y + row) * blockDim.x + col;
    if (id < n) {
        tile[row][col] = data[id];
    }
    __syncthreads();
    if (id < n) {
        data[id] = tile[row][col] * 2.0f;
    }
}

__global__ void histogramGlobal(int *data, int *hist, int n, int bins) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < n) {
        int bin = data[id] % bins;
        atomicAdd(&hist[bin], 1);
    }
}

__global__ void histogramShared(int *data, int *hist, int n, int bins) {
    extern __shared__ int localHist[];
    int tid = threadIdx.x;

    for (int i = tid; i < bins; i = i + blockDim.x) {
        localHist[i] = 0;
    }
    __syncthreads();

    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < n) {
        int bin = data[id] % bins;
        atomicAdd(&localHist[bin], 1);
    }
    __syncthreads();

    for (int i = tid; i < bins; i = i + blockDim.x) {
        atomicAdd(&hist[i], localHist[i]);
    }
}

void runHistogramBenchmark() {
    printf("\n--- Histogram with Shared Memory ---\n");
    int n = 1 << 20;
    int bins = 256;

    int *h_data = (int*)malloc(n * sizeof(int));
    for (int i = 0; i < n; i++) h_data[i] = i % bins;

    int *d_data, *d_hist_global, *d_hist_shared;
    CUDA_CHECK(cudaMalloc((void**)&d_data, n * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&d_hist_global, bins * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&d_hist_shared, bins * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_data, h_data, n * sizeof(int), cudaMemcpyHostToDevice));

    int blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;

    cudaEvent_t t0, t1;
    cudaEventCreate(&t0);
    cudaEventCreate(&t1);

    cudaMemset(d_hist_global, 0, bins * sizeof(int));
    cudaEventRecord(t0);
    histogramGlobal<<<blocks, BLOCK_SIZE>>>(d_data, d_hist_global, n, bins);
    cudaEventRecord(t1);
    cudaEventSynchronize(t1);
    float global_ms;
    cudaEventElapsedTime(&global_ms, t0, t1);

    cudaMemset(d_hist_shared, 0, bins * sizeof(int));
    cudaEventRecord(t0);
    histogramShared<<<blocks, BLOCK_SIZE, bins * sizeof(int)>>>(d_data, d_hist_shared, n, bins);
    cudaEventRecord(t1);
    cudaEventSynchronize(t1);
    float shared_ms;
    cudaEventElapsedTime(&shared_ms, t0, t1);

    printf("Global memory histogram: %.3f ms\n", global_ms);
    printf("Shared memory histogram: %.3f ms\n", shared_ms);
    printf("Speedup: %.2fx\n", global_ms / shared_ms);

    free(h_data);
    cudaFree(d_data); cudaFree(d_hist_global); cudaFree(d_hist_shared);
    cudaEventDestroy(t0); cudaEventDestroy(t1);
}

int main() {
    printf("=== EX02: Memory Hierarchy ===\n");
    runReductionBenchmark();
    runBankConflictBenchmark();
    runHistogramBenchmark();
    return 0;
}
