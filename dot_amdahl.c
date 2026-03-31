// Question 3: Distributed Dot Product & Amdahl's Law
// Dot product of two 500-million-element vectors, each process generates its own chunk.
// Uses MPI_Bcast (for multiplier) + MPI_Reduce (for final sum).
// To compile: mpicc -O2 -o dot_amdahl dot_amdahl.c
// To run:     mpirun -np 1 ./dot_amdahl   (then 2, 4, 8)

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#define TOTAL_SIZE 500000000LL   // 500 million elements

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Step 1: Rank 0 sets a scaling multiplier; broadcast to all
    double multiplier = 1.0;
    if (rank == 0) {
        multiplier = 3.0;   // Change this to experiment
        printf("Processes: %d | Multiplier: %.1f\n", size, multiplier);
    }
    MPI_Bcast(&multiplier, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Step 2: Each process generates only its local chunk (saves memory)
    long long chunk = TOTAL_SIZE / size;
    long long start = (long long)rank * chunk;

    double* A = (double*)malloc(chunk * sizeof(double));
    double* B = (double*)malloc(chunk * sizeof(double));
    if (!A || !B) {
        fprintf(stderr, "Rank %d: malloc failed\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    for (long long i = 0; i < chunk; i++) {
        A[i] = 1.0;
        B[i] = 2.0 * multiplier;
    }

    // Step 3: Time computation + communication together
    MPI_Barrier(MPI_COMM_WORLD);
    double t_start = MPI_Wtime();

    // Local dot product
    double local_dot = 0.0;
    for (long long i = 0; i < chunk; i++) local_dot += A[i] * B[i];

    // Step 4: Global reduction
    double final_result = 0.0;
    MPI_Reduce(&local_dot, &final_result, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    double t_end = MPI_Wtime();
    double elapsed = t_end - t_start;

    if (rank == 0) {
        printf("Dot product result : %.2e\n", final_result);
        printf("Expected           : %.2e\n", (double)TOTAL_SIZE * 1.0 * 2.0 * multiplier);
        printf("Time               : %.4f seconds\n", elapsed);
        printf("(Record this time for speedup/efficiency table)\n");
    }

    free(A);
    free(B);
    MPI_Finalize();
    return 0;
}
