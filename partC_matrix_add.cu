#include <stdio.h>
#include <cuda_runtime.h>

#define N 512
#define BLOCK_SIZE 16

__global__ void matAddKernel(int *A, int *B, int *C) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N)
        C[row * N + col] = A[row * N + col] + B[row * N + col];
}

int main() {
    int size = N * N * sizeof(int);
    int *h_A = (int *)malloc(size);
    int *h_B = (int *)malloc(size);
    int *h_C = (int *)malloc(size);

    for (int i = 0; i < N * N; i++) {
        h_A[i] = 1;
        h_B[i] = 2;
    }

    int *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid(N / BLOCK_SIZE, N / BLOCK_SIZE);
    matAddKernel<<<grid, block>>>(d_A, d_B, d_C);

    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    printf("C[0][0] = %d\n", h_C[0]);
    printf("C[N-1][N-1] = %d\n", h_C[N * N - 1]);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);
    return 0;
}
