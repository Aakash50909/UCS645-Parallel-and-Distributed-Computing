#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

#define CUDA_CHECK(x) { cudaError_t e = x; if(e != cudaSuccess){ printf("CUDA error %s line %d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); exit(1); } }

__global__ void sigmoidKernel(float *input, float *output, int n) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < n) {
        output[id] = 1.0f / (1.0f + expf(-input[id]));
    }
}

__global__ void tanhKernel(float *input, float *output, int n) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < n) {
        float ep = expf(input[id]);
        float en = expf(-input[id]);
        output[id] = (ep - en) / (ep + en);
    }
}

__global__ void leakyReluKernel(float *input, float *output, float alpha, int n) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < n) {
        if (input[id] > 0.0f) {
            output[id] = input[id];
        } else {
            output[id] = alpha * input[id];
        }
    }
}

__global__ void reluBackwardKernel(float *input, float *gradOut, float *gradIn, int n) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < n) {
        if (input[id] > 0.0f) {
            gradIn[id] = gradOut[id];
        } else {
            gradIn[id] = 0.0f;
        }
    }
}

__global__ void bceLossKernel(float *preds, float *targets, float *losses, int n) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < n) {
        float p = preds[id];
        if (p < 1e-7f) p = 1e-7f;
        if (p > 1.0f - 1e-7f) p = 1.0f - 1e-7f;
        float t = targets[id];
        losses[id] = -(t * logf(p) + (1.0f - t) * logf(1.0f - p));
    }
}

__global__ void crossEntropyKernel(float *logits, int *labels, float *losses, int n, int numClasses) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < n) {
        float *row = logits + id * numClasses;
        float maxVal = row[0];
        for (int c = 1; c < numClasses; c++) {
            if (row[c] > maxVal) maxVal = row[c];
        }
        float sumExp = 0.0f;
        for (int c = 0; c < numClasses; c++) {
            sumExp = sumExp + expf(row[c] - maxVal);
        }
        float logSumExp = logf(sumExp) + maxVal;
        losses[id] = logSumExp - row[labels[id]];
    }
}

__global__ void crossEntropyGradKernel(float *logits, int *labels, float *gradLogits, int n, int numClasses) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < n) {
        float *row = logits + id * numClasses;
        float *gradRow = gradLogits + id * numClasses;
        float maxVal = row[0];
        for (int c = 1; c < numClasses; c++) {
            if (row[c] > maxVal) maxVal = row[c];
        }
        float sumExp = 0.0f;
        for (int c = 0; c < numClasses; c++) {
            sumExp = sumExp + expf(row[c] - maxVal);
        }
        for (int c = 0; c < numClasses; c++) {
            float softmax = expf(row[c] - maxVal) / sumExp;
            gradRow[c] = softmax - (c == labels[id] ? 1.0f : 0.0f);
        }
    }
}

__global__ void adamUpdateKernel(float *params, float *grads, float *m, float *v,
                                  float lr, float beta1, float beta2, float eps,
                                  int t, int n) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < n) {
        float g = grads[id];
        m[id] = beta1 * m[id] + (1.0f - beta1) * g;
        v[id] = beta2 * v[id] + (1.0f - beta2) * g * g;
        float mHat = m[id] / (1.0f - powf(beta1, (float)t));
        float vHat = v[id] / (1.0f - powf(beta2, (float)t));
        params[id] = params[id] - lr * mHat / (sqrtf(vHat) + eps);
    }
}

void benchmarkActivation(const char *name, int n) {
    size_t bytes = (size_t)n * sizeof(float);
    float *d_in, *d_out;
    CUDA_CHECK(cudaMalloc((void**)&d_in, bytes));
    CUDA_CHECK(cudaMalloc((void**)&d_out, bytes));

    int threads = 256;
    int blocks = (n + threads - 1) / threads;

    cudaEvent_t t0, t1;
    cudaEventCreate(&t0);
    cudaEventCreate(&t1);

    int iters = 100;
    cudaEventRecord(t0);
    for (int i = 0; i < iters; i++) {
        if (strcmp(name, "sigmoid") == 0) sigmoidKernel<<<blocks, threads>>>(d_in, d_out, n);
        else if (strcmp(name, "tanh") == 0) tanhKernel<<<blocks, threads>>>(d_in, d_out, n);
        else if (strcmp(name, "leaky_relu") == 0) leakyReluKernel<<<blocks, threads>>>(d_in, d_out, 0.01f, n);
        else if (strcmp(name, "relu_backward") == 0) reluBackwardKernel<<<blocks, threads>>>(d_in, d_out, d_in, n);
    }
    cudaEventRecord(t1);
    cudaEventSynchronize(t1);
    float ms;
    cudaEventElapsedTime(&ms, t0, t1);

    float avgMs = ms / iters;
    double bw = (double)bytes * 2 / (avgMs / 1000.0) / 1e9;
    printf("%-15s  time: %.4f ms  bandwidth: %.1f GB/s\n", name, avgMs, bw);

    cudaFree(d_in);
    cudaFree(d_out);
    cudaEventDestroy(t0);
    cudaEventDestroy(t1);
}

int main() {
    printf("=== EX03: ML Primitives ===\n");
    int n = 10000000;

    printf("\n--- Activation Function Benchmarks (N = 10^7) ---\n");
    benchmarkActivation("sigmoid", n);
    benchmarkActivation("tanh", n);
    benchmarkActivation("leaky_relu", n);
    benchmarkActivation("relu_backward", n);

    printf("\n--- Correctness Checks ---\n");
    int testN = 5;
    float h_in[5] = {-2.0f, -1.0f, 0.0f, 1.0f, 2.0f};
    float h_out[5];
    float *d_in, *d_out;
    cudaMalloc((void**)&d_in, testN * sizeof(float));
    cudaMalloc((void**)&d_out, testN * sizeof(float));
    cudaMemcpy(d_in, h_in, testN * sizeof(float), cudaMemcpyHostToDevice);

    sigmoidKernel<<<1, testN>>>(d_in, d_out, testN);
    cudaMemcpy(h_out, d_out, testN * sizeof(float), cudaMemcpyDeviceToHost);
    printf("Sigmoid(-2,-1,0,1,2): ");
    for (int i = 0; i < testN; i++) printf("%.4f ", h_out[i]);
    printf("\n");

    tanhKernel<<<1, testN>>>(d_in, d_out, testN);
    cudaMemcpy(h_out, d_out, testN * sizeof(float), cudaMemcpyDeviceToHost);
    printf("Tanh(-2,-1,0,1,2):    ");
    for (int i = 0; i < testN; i++) printf("%.4f ", h_out[i]);
    printf("\n");

    leakyReluKernel<<<1, testN>>>(d_in, d_out, 0.01f, testN);
    cudaMemcpy(h_out, d_out, testN * sizeof(float), cudaMemcpyDeviceToHost);
    printf("LeakyReLU(-2,-1,0,1,2): ");
    for (int i = 0; i < testN; i++) printf("%.4f ", h_out[i]);
    printf("\n");

    printf("\n--- Adam Optimizer Test (10 steps) ---\n");
    int paramN = 1000;
    float *d_params, *d_grads, *d_m, *d_v;
    cudaMalloc((void**)&d_params, paramN * sizeof(float));
    cudaMalloc((void**)&d_grads, paramN * sizeof(float));
    cudaMalloc((void**)&d_m, paramN * sizeof(float));
    cudaMalloc((void**)&d_v, paramN * sizeof(float));
    cudaMemset(d_params, 0, paramN * sizeof(float));
    cudaMemset(d_m, 0, paramN * sizeof(float));
    cudaMemset(d_v, 0, paramN * sizeof(float));

    float h_grads[1000];
    for (int i = 0; i < paramN; i++) h_grads[i] = 0.1f;
    cudaMemcpy(d_grads, h_grads, paramN * sizeof(float), cudaMemcpyHostToDevice);

    int adamBlocks = (paramN + 255) / 256;
    for (int step = 1; step <= 10; step++) {
        adamUpdateKernel<<<adamBlocks, 256>>>(d_params, d_grads, d_m, d_v,
                                              0.001f, 0.9f, 0.999f, 1e-8f, step, paramN);
    }

    float h_params[5];
    cudaMemcpy(h_params, d_params, 5 * sizeof(float), cudaMemcpyDeviceToHost);
    printf("First 5 params after 10 Adam steps: ");
    for (int i = 0; i < 5; i++) printf("%.6f ", h_params[i]);
    printf("\n");

    cudaFree(d_in); cudaFree(d_out);
    cudaFree(d_params); cudaFree(d_grads); cudaFree(d_m); cudaFree(d_v);

    return 0;
}
