#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

int main() {
    long n = 65536;
    double a = 2.0;
    double *x = malloc(n * sizeof(double));
    double *y = malloc(n * sizeof(double));
    double start, end;
    int i;

    for (i = 0; i < n; i++) {
        x[i] = 1.0;
        y[i] = 2.0;
    }

    start = omp_get_wtime();

    #pragma omp parallel for
    for (i = 0; i < n; i++) {
        x[i] = a * x[i] + y[i];
    }

    end = omp_get_wtime();

    printf("Time taken: %f seconds\n", end - start);

    free(x);
    free(y);
    return 0;
}

