// Question 2: The Broadcast Race — MyBcast (linear) vs MPI_Bcast (tree-optimized)
// Array of 10 million doubles (~80 MB) broadcast from Rank 0.
// To compile: mpicc -O2 -o bcast_race bcast_race.c
// To run:     mpirun -np 4 ./bcast_race   (try 2, 4, 8, 16)

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#define ARRAY_SIZE 10000000   // 10 million doubles ≈ 80 MB

// Part A: Linear broadcast — rank 0 loops through MPI_Send to every other rank
void MyBcast(double* buf, int count, int rank, int size) {
    if (rank == 0) {
        for (int dest = 1; dest < size; dest++)
            MPI_Send(buf, count, MPI_DOUBLE, dest, 0, MPI_COMM_WORLD);
    } else {
        MPI_Recv(buf, count, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    double* buf = (double*)malloc(ARRAY_SIZE * sizeof(double));

    // Initialize on rank 0
    if (rank == 0)
        for (int i = 0; i < ARRAY_SIZE; i++) buf[i] = (double)i * 0.5;

    // ---- Part A: MyBcast (linear for-loop) ----
    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();
    MyBcast(buf, ARRAY_SIZE, rank, size);
    MPI_Barrier(MPI_COMM_WORLD);
    double t_mybcast = MPI_Wtime() - t0;

    // Re-initialize for a fair comparison
    if (rank == 0)
        for (int i = 0; i < ARRAY_SIZE; i++) buf[i] = (double)i * 0.5;

    // ---- Part B: MPI_Bcast (library-optimized, typically tree-based) ----
    MPI_Barrier(MPI_COMM_WORLD);
    double t1 = MPI_Wtime();
    MPI_Bcast(buf, ARRAY_SIZE, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);
    double t_mpibcast = MPI_Wtime() - t1;

    if (rank == 0) {
        printf("Processes: %d\n", size);
        printf("MyBcast  (linear): %.4f seconds\n", t_mybcast);
        printf("MPI_Bcast (tree) : %.4f seconds\n", t_mpibcast);
        printf("Speedup of MPI_Bcast over MyBcast: %.2fx\n", t_mybcast / t_mpibcast);
        printf("\nAnalysis:\n");
        printf("  MyBcast is O(p) — rank 0 sends to each process sequentially.\n");
        printf("  MPI_Bcast is O(log p) — tree-based, parallelizes intermediate sends.\n");
    }

    free(buf);
    MPI_Finalize();
    return 0;
}
