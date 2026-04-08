#include <stdio.h>
#include <cuda_runtime.h>

#define N 1024
#define BLOCK_SIZE 256

__global__ void sumKernel(float *input, float *output) {
    __shared__ float temp[BLOCK_SIZE];
    int i = threadIdx.x;
    temp[i] = input[blockIdx.x * blockDim.x + i];
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (i < stride)
            temp[i] += temp[i + stride];
        __syncthreads();
    }

    if (i == 0)
        output[blockIdx.x] = temp[0];
}

int main() {
    float h_input[N], h_output[N / BLOCK_SIZE];
    float *d_input, *d_output;

    for (int i = 0; i < N; i++)
        h_input[i] = 1.0f;

    cudaMalloc(&d_input, N * sizeof(float));
    cudaMalloc(&d_output, (N / BLOCK_SIZE) * sizeof(float));

    cudaMemcpy(d_input, h_input, N * sizeof(float), cudaMemcpyHostToDevice);

    dim3 block(BLOCK_SIZE);
    dim3 grid(N / BLOCK_SIZE);
    sumKernel<<<grid, block>>>(d_input, d_output);

    cudaMemcpy(h_output, d_output, (N / BLOCK_SIZE) * sizeof(float), cudaMemcpyDeviceToHost);

    float total = 0;
    for (int i = 0; i < N / BLOCK_SIZE; i++)
        total += h_output[i];

    printf("Sum = %.2f\n", total);

    cudaFree(d_input);
    cudaFree(d_output);
    return 0;
}
