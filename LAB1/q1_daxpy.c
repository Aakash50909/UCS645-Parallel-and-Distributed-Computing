#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#define N 65536 // 2^16
#define A 2.0   // Scalar 'a'

int main() {
    double *x = (double*)malloc(N * sizeof(double));
    double *y = (double*)malloc(N * sizeof(double));
    
    // Initialize vectors
    for (int i = 0; i < N; i++) {
        x[i] = 1.0;
        y[i] = 2.0;
    }

    double start = omp_get_wtime();

    // Parallel DAXPY Loop
    #pragma omp parallel for
    for (int i = 0; i < N; i++) {
        x[i] = A * x[i] + y[i];
    }

    double end = omp_get_wtime();
    
    printf("Time taken: %f seconds\n", end - start);
    
    free(x);
    free(y);
    return 0;
}
