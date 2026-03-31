// Question 1: DAXPY using MPI
// Operation: X[i] = a*X[i] + Y[i]  for vectors of size 2^16
// Measures speedup of MPI vs uniprocessor implementation.
// To compile: mpicc -O2 -o daxpy daxpy.c
// To run:     mpirun -np 4 ./daxpy

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#define N (1 << 16)   // 2^16 = 65536 elements
#define ALPHA 2.5     // Scalar multiplier

// Sequential DAXPY (runs only on rank 0 for baseline timing)
static void daxpy_seq(double* X, const double* Y, double a, int n) {
    for (int i = 0; i < n; i++) X[i] = a * X[i] + Y[i];
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int chunk = N / size;   // Each process handles this many elements

    double* X_full = NULL;
    double* Y_full = NULL;
    double* X_local = (double*)malloc(chunk * sizeof(double));
    double* Y_local = (double*)malloc(chunk * sizeof(double));
    double seq_time = 0.0;

    // --- Baseline: Sequential timing on rank 0 ---
    if (rank == 0) {
        X_full = (double*)malloc(N * sizeof(double));
        Y_full = (double*)malloc(N * sizeof(double));
        for (int i = 0; i < N; i++) { X_full[i] = (double)i; Y_full[i] = (double)(N - i); }

        // Sequential run
        double* Xs = (double*)malloc(N * sizeof(double));
        double* Ys = (double*)malloc(N * sizeof(double));
        for (int i = 0; i < N; i++) { Xs[i] = X_full[i]; Ys[i] = Y_full[i]; }

        double t0 = MPI_Wtime();
        daxpy_seq(Xs, Ys, ALPHA, N);
        seq_time = MPI_Wtime() - t0;
        printf("Sequential time: %.6f seconds\n", seq_time);
        free(Xs); free(Ys);
    }

    // --- Parallel MPI DAXPY ---
    MPI_Barrier(MPI_COMM_WORLD);   // Sync before parallel timing
    double par_start = MPI_Wtime();

    // Distribute X and Y to all processes
    MPI_Scatter(X_full, chunk, MPI_DOUBLE, X_local, chunk, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatter(Y_full, chunk, MPI_DOUBLE, Y_local, chunk, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Each process computes its local DAXPY chunk
    for (int i = 0; i < chunk; i++) X_local[i] = ALPHA * X_local[i] + Y_local[i];

    // Gather results back to rank 0
    MPI_Gather(X_local, chunk, MPI_DOUBLE, X_full, chunk, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    double par_time = MPI_Wtime() - par_start;

    if (rank == 0) {
        printf("Parallel time (%d procs): %.6f seconds\n", size, par_time);
        printf("Speedup: %.2fx\n", seq_time / par_time);
        printf("Efficiency: %.2f%%\n", (seq_time / par_time) / size * 100.0);
        // Quick sanity check on first element: a*0 + N = N
        printf("Sanity check X[0] = %.1f (expected %.1f)\n", X_full[0], (double)N);
        free(X_full);
        free(Y_full);
    }

    free(X_local);
    free(Y_local);
    MPI_Finalize();
    return 0;
}
