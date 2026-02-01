#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#define N 1000

double a[N][N];
double b[N][N];
double c[N][N];

int main() {
    int i, j, k;
    double start, end;

    for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
            a[i][j] = 1.0;
            b[i][j] = 1.0;
        }
    }

    start = omp_get_wtime();

    #pragma omp parallel for
    for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
            for (k = 0; k < N; k++) {
                c[i][j] += a[i][k] * b[k][j];
            }
        }
    }

    end = omp_get_wtime();
    printf("1D Time: %f seconds\n", end - start);

    for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
            c[i][j] = 0.0;
        }
    }

    start = omp_get_wtime();

    #pragma omp parallel for collapse(2)
    for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
            for (k = 0; k < N; k++) {
                c[i][j] += a[i][k] * b[k][j];
            }
        }
    }

    end = omp_get_wtime();
    printf("2D Time: %f seconds\n", end - start);

    return 0;
}
