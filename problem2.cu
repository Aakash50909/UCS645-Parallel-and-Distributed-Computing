#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>

#define N 1000

void merge(int *arr, int left, int mid, int right) {
    int n1 = mid - left + 1;
    int n2 = right - mid;

    int *L = (int*)malloc(n1 * sizeof(int));
    int *R = (int*)malloc(n2 * sizeof(int));

    for (int i = 0; i < n1; i++) L[i] = arr[left + i];
    for (int j = 0; j < n2; j++) R[j] = arr[mid + 1 + j];

    int i = 0, j = 0, k = left;
    while (i < n1 && j < n2) {
        if (L[i] <= R[j]) {
            arr[k] = L[i];
            i++;
        } else {
            arr[k] = R[j];
            j++;
        }
        k++;
    }
    while (i < n1) { arr[k] = L[i]; i++; k++; }
    while (j < n2) { arr[k] = R[j]; j++; k++; }

    free(L);
    free(R);
}

void pipelinedMergeSort(int *arr, int n) {
    for (int width = 1; width < n; width = width * 2) {
        for (int left = 0; left < n - width; left = left + 2 * width) {
            int mid = left + width - 1;
            int right = left + 2 * width - 1;
            if (right >= n) right = n - 1;
            merge(arr, left, mid, right);
        }
    }
}

__device__ void deviceMerge(int *arr, int *temp, int left, int mid, int right) {
    int i = left, j = mid + 1, k = left;
    while (i <= mid && j <= right) {
        if (arr[i] <= arr[j]) {
            temp[k] = arr[i];
            i++;
        } else {
            temp[k] = arr[j];
            j++;
        }
        k++;
    }
    while (i <= mid) { temp[k] = arr[i]; i++; k++; }
    while (j <= right) { temp[k] = arr[j]; j++; k++; }
    for (int x = left; x <= right; x++) arr[x] = temp[x];
}

__global__ void cudaMergeSortKernel(int *arr, int *temp, int n, int width) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    int left = id * 2 * width;
    if (left >= n) return;
    int mid = left + width - 1;
    int right = left + 2 * width - 1;
    if (mid >= n) return;
    if (right >= n) right = n - 1;
    deviceMerge(arr, temp, left, mid, right);
}

void cudaMergeSortHost(int *h_arr, int n) {
    int *d_arr, *d_temp;
    cudaMalloc((void**)&d_arr, n * sizeof(int));
    cudaMalloc((void**)&d_temp, n * sizeof(int));

    cudaMemcpy(d_arr, h_arr, n * sizeof(int), cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;

    for (int width = 1; width < n; width = width * 2) {
        int numPairs = (n + 2 * width - 1) / (2 * width);
        int numBlocks = (numPairs + threadsPerBlock - 1) / threadsPerBlock;
        cudaMergeSortKernel<<<numBlocks, threadsPerBlock>>>(d_arr, d_temp, n, width);
        cudaDeviceSynchronize();
    }

    cudaMemcpy(h_arr, d_arr, n * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_arr);
    cudaFree(d_temp);
}

int main() {
    int arr1[N], arr2[N];
    srand(42);
    for (int i = 0; i < N; i++) {
        int val = rand() % 10000;
        arr1[i] = val;
        arr2[i] = val;
    }

    clock_t start1 = clock();
    pipelinedMergeSort(arr1, N);
    clock_t end1 = clock();
    double time1 = ((double)(end1 - start1)) / CLOCKS_PER_SEC * 1000.0;

    cudaEvent_t start2, end2;
    cudaEventCreate(&start2);
    cudaEventCreate(&end2);
    cudaEventRecord(start2);

    cudaMergeSortHost(arr2, N);

    cudaEventRecord(end2);
    cudaEventSynchronize(end2);
    float time2 = 0;
    cudaEventElapsedTime(&time2, start2, end2);

    printf("Pipelining CPU merge sort time: %.3f ms\n", time1);
    printf("CUDA GPU merge sort time:       %.3f ms\n", time2);
    printf("Speedup (CPU/GPU): %.2fx\n", time1 / time2);

    int correct = 1;
    for (int i = 0; i < N - 1; i++) {
        if (arr1[i] > arr1[i+1]) { correct = 0; break; }
        if (arr2[i] > arr2[i+1]) { correct = 0; break; }
    }
    printf("Both sorts correct: %s\n", correct ? "YES" : "NO");

    return 0;
}
