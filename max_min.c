// Exercise 3: Finding Global Maximum and Minimum
// Each process generates 10 random numbers (0-1000).
// Uses MPI_MAXLOC / MPI_MINLOC to also report which rank held the extreme value.
// To compile: mpicc -o max_min max_min.c
// To run:     mpirun -np 4 ./max_min

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define LOCAL_COUNT 10

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Seed differently per process
    srand(time(NULL) + rank * 1000);

    int local_nums[LOCAL_COUNT];
    int local_max = 0, local_min = 1001;

    printf("Process %d generated: ", rank);
    for (int i = 0; i < LOCAL_COUNT; i++) {
        local_nums[i] = rand() % 1001;
        printf("%d ", local_nums[i]);
        if (local_nums[i] > local_max) local_max = local_nums[i];
        if (local_nums[i] < local_min) local_min = local_nums[i];
    }
    printf("\n");
    printf("Process %d: local_max=%d, local_min=%d\n", rank, local_max, local_min);

    // MPI_MAXLOC / MPI_MINLOC use a pair {value, rank}
    struct { int val; int rank; } send_max, send_min, recv_max, recv_min;
    send_max.val = local_max; send_max.rank = rank;
    send_min.val = local_min; send_min.rank = rank;

    MPI_Reduce(&send_max, &recv_max, 1, MPI_2INT, MPI_MAXLOC, 0, MPI_COMM_WORLD);
    MPI_Reduce(&send_min, &recv_min, 1, MPI_2INT, MPI_MINLOC, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("\n========================================\n");
        printf("Global Maximum: %d (from Process %d)\n", recv_max.val, recv_max.rank);
        printf("Global Minimum: %d (from Process %d)\n", recv_min.val, recv_min.rank);
        printf("========================================\n");
    }

    MPI_Finalize();
    return 0;
}
