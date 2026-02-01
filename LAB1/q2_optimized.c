#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#define N 1000 // 1000x1000 matrix

double a[N][N];
double b[N][N];
double c[N][N];
double b_transposed[N][N]; // Extra memory for speed!

int main() {
    int i, j, k;
    double start, end;

    // 1. Initialize
    for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
            a[i][j] = 1.0;
            b[i][j] = 1.0;
            c[i][j] = 0.0;
        }
    }

    start = omp_get_wtime();

    // 2. Transpose Matrix B (The Secret Sauce)
    // We flip B so we can read it row-by-row instead of column-by-column
    #pragma omp parallel for collapse(2)
    for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
            b_transposed[j][i] = b[i][j];
        }
    }

    // 3. Multiply using the Transposed Matrix
    #pragma omp parallel for collapse(2)
    for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
            double sum = 0.0;
            for (k = 0; k < N; k++) {
                // Now we access b_transposed[j][k] which is ROW-WISE (Fast!)
                sum += a[i][k] * b_transposed[j][k];
            }
            c[i][j] = sum;
        }
    }

    end = omp_get_wtime();
    printf("Optimized (Transposed) Time: %f seconds\n", end - start);

    return 0;
}
