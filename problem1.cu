#include <stdio.h>
#include <cuda_runtime.h>

#define N 1024

__global__ void sumIterative(int *input, long long *output) {
    int id = threadIdx.x;
    if (id == 0) {
        long long total = 0;
        for (int i = 0; i < N; i++) {
            total = total + input[i];
        }
        output[0] = total;
    }
}

__global__ void sumFormula(int *input, long long *output) {
    int id = threadIdx.x;
    if (id == 1) {
        long long n = N;
        output[1] = n * (n + 1) / 2;
    }
}

int main() {
    int h_input[N];
    long long h_output[2];

    for (int i = 0; i < N; i++) {
        h_input[i] = i + 1;
    }

    int *d_input;
    long long *d_output;

    cudaMalloc((void**)&d_input, N * sizeof(int));
    cudaMalloc((void**)&d_output, 2 * sizeof(long long));

    cudaMemcpy(d_input, h_input, N * sizeof(int), cudaMemcpyHostToDevice);

    sumIterative<<<1, N>>>(d_input, d_output);
    sumFormula<<<1, N>>>(d_input, d_output);

    cudaMemcpy(h_output, d_output, 2 * sizeof(long long), cudaMemcpyDeviceToHost);

    printf("Sum using iterative method: %lld\n", h_output[0]);
    printf("Sum using formula method:   %lld\n", h_output[1]);

    cudaFree(d_input);
    cudaFree(d_output);

    return 0;
}
