#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#define CUDA_CHECK(x) { cudaError_t e = x; if(e != cudaSuccess){ printf("CUDA error %s line %d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); exit(1); } }

#define TILE 16

__global__ void naiveMatMul(float *A, float *B, float *C, int M, int K, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            sum = sum + A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

__global__ void tiledMatMul(float *A, float *B, float *C, int M, int K, int N) {
    __shared__ float tileA[TILE][TILE];
    __shared__ float tileB[TILE][TILE];

    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;

    float sum = 0.0f;

    int numTiles = (K + TILE - 1) / TILE;
    for (int t = 0; t < numTiles; t++) {
        int aCol = t * TILE + threadIdx.x;
        int bRow = t * TILE + threadIdx.y;

        if (row < M && aCol < K) {
            tileA[threadIdx.y][threadIdx.x] = A[row * K + aCol];
        } else {
            tileA[threadIdx.y][threadIdx.x] = 0.0f;
        }

        if (bRow < K && col < N) {
            tileB[threadIdx.y][threadIdx.x] = B[bRow * N + col];
        } else {
            tileB[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads();

        for (int k = 0; k < TILE; k++) {
            sum = sum + tileA[threadIdx.y][k] * tileB[k][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

void benchmarkGEMM() {
    printf("\n--- GEMM Benchmark ---\n");
    printf("%-8s %-15s %-15s %-15s %-15s %-15s\n",
           "Size", "Naive(ms)", "Tiled(ms)", "cuBLAS(ms)", "Naive GFLOPS", "Tiled GFLOPS");

    cublasHandle_t handle;
    cublasCreate(&handle);

    int sizes[] = {128, 256, 512, 1024, 2048};
    int numSizes = 5;

    for (int s = 0; s < numSizes; s++) {
        int sz = sizes[s];
        int M = sz, K = sz, N = sz;
        size_t bytesA = (size_t)M * K * sizeof(float);
        size_t bytesB = (size_t)K * N * sizeof(float);
        size_t bytesC = (size_t)M * N * sizeof(float);

        float *d_A, *d_B, *d_C;
        CUDA_CHECK(cudaMalloc((void**)&d_A, bytesA));
        CUDA_CHECK(cudaMalloc((void**)&d_B, bytesB));
        CUDA_CHECK(cudaMalloc((void**)&d_C, bytesC));

        cudaEvent_t t0, t1;
        cudaEventCreate(&t0);
        cudaEventCreate(&t1);

        dim3 threads(TILE, TILE);
        dim3 naiveBlocks((N + 15) / 16, (M + 15) / 16);

        cudaEventRecord(t0);
        naiveMatMul<<<naiveBlocks, threads>>>(d_A, d_B, d_C, M, K, N);
        cudaEventRecord(t1);
        cudaEventSynchronize(t1);
        float naive_ms;
        cudaEventElapsedTime(&naive_ms, t0, t1);

        cudaEventRecord(t0);
        tiledMatMul<<<naiveBlocks, threads>>>(d_A, d_B, d_C, M, K, N);
        cudaEventRecord(t1);
        cudaEventSynchronize(t1);
        float tiled_ms;
        cudaEventElapsedTime(&tiled_ms, t0, t1);

        float alpha = 1.0f, beta = 0.0f;
        cudaEventRecord(t0);
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K,
                    &alpha, d_B, N, d_A, K, &beta, d_C, N);
        cudaEventRecord(t1);
        cudaEventSynchronize(t1);
        float cublas_ms;
        cudaEventElapsedTime(&cublas_ms, t0, t1);

        double flops = 2.0 * M * N * K;
        double naive_gflops = flops / (naive_ms / 1000.0) / 1e9;
        double tiled_gflops = flops / (tiled_ms / 1000.0) / 1e9;

        printf("%-8d %-15.3f %-15.3f %-15.3f %-15.1f %-15.1f\n",
               sz, naive_ms, tiled_ms, cublas_ms, naive_gflops, tiled_gflops);

        cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
        cudaEventDestroy(t0); cudaEventDestroy(t1);
    }

    cublasDestroy(handle);
}

__global__ void maxPool2x2(float *input, float *output, int N, int C, int H, int W) {
    int outH = H / 2;
    int outW = W / 2;
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C * outH * outW;
    if (id >= total) return;

    int ow = id % outW;
    int oh = (id / outW) % outH;
    int c = (id / outW / outH) % C;
    int n = id / outW / outH / C;

    float maxVal = -1e30f;
    for (int dh = 0; dh < 2; dh++) {
        for (int dw = 0; dw < 2; dw++) {
            int ih = oh * 2 + dh;
            int iw = ow * 2 + dw;
            float val = input[((n * C + c) * H + ih) * W + iw];
            if (val > maxVal) maxVal = val;
        }
    }
    output[id] = maxVal;
}

__global__ void batchNormInference(float *input, float *gamma, float *beta,
                                    float *mean, float *var, float *output,
                                    int N, int C, int H, int W, float eps) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C * H * W;
    if (id >= total) return;

    int c = (id / (H * W)) % C;
    float x = input[id];
    float normalized = (x - mean[c]) / sqrtf(var[c] + eps);
    output[id] = gamma[c] * normalized + beta[c];
}

void benchmarkCNNLayers() {
    printf("\n--- CNN Layer Benchmarks [32, 64, 14, 14] ---\n");

    int batch = 32, channels = 64, H = 14, W = 14;
    int total = batch * channels * H * W;
    size_t bytes = (size_t)total * sizeof(float);

    float *d_input, *d_output;
    CUDA_CHECK(cudaMalloc((void**)&d_input, bytes));
    CUDA_CHECK(cudaMalloc((void**)&d_output, bytes));

    cudaEvent_t t0, t1;
    cudaEventCreate(&t0);
    cudaEventCreate(&t1);

    int threads = 256;
    int outTotal = batch * channels * (H/2) * (W/2);
    int blocks = (outTotal + threads - 1) / threads;

    cudaEventRecord(t0);
    for (int i = 0; i < 100; i++) maxPool2x2<<<blocks, threads>>>(d_input, d_output, batch, channels, H, W);
    cudaEventRecord(t1);
    cudaEventSynchronize(t1);
    float pool_ms;
    cudaEventElapsedTime(&pool_ms, t0, t1);
    printf("MaxPool 2x2:       %.4f ms\n", pool_ms / 100.0f);

    float *d_gamma, *d_beta, *d_mean, *d_var;
    cudaMalloc((void**)&d_gamma, channels * sizeof(float));
    cudaMalloc((void**)&d_beta, channels * sizeof(float));
    cudaMalloc((void**)&d_mean, channels * sizeof(float));
    cudaMalloc((void**)&d_var, channels * sizeof(float));

    int bnBlocks = (total + threads - 1) / threads;
    cudaEventRecord(t0);
    for (int i = 0; i < 100; i++) {
        batchNormInference<<<bnBlocks, threads>>>(d_input, d_gamma, d_beta,
                                                   d_mean, d_var, d_output,
                                                   batch, channels, H, W, 1e-5f);
    }
    cudaEventRecord(t1);
    cudaEventSynchronize(t1);
    float bn_ms;
    cudaEventElapsedTime(&bn_ms, t0, t1);
    printf("BatchNorm Inference: %.4f ms\n", bn_ms / 100.0f);

    cudaFree(d_input); cudaFree(d_output);
    cudaFree(d_gamma); cudaFree(d_beta); cudaFree(d_mean); cudaFree(d_var);
    cudaEventDestroy(t0); cudaEventDestroy(t1);
}

int main() {
    printf("=== EX04: CNN Layers ===\n");
    benchmarkGEMM();
    benchmarkCNNLayers();
    return 0;
}
