// Exercise 2: Parallel Array Sum
// Array of 1..100 distributed via MPI_Scatter, summed locally,
// then combined with MPI_Reduce. Expected global sum = 5050.
// To compile: mpicc -o array_sum array_sum.c
// To run:     mpirun -np 4 ./array_sum

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#define ARRAY_SIZE 100

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int chunk_size = ARRAY_SIZE / size;
    int* array     = NULL;
    int* local     = (int*)malloc(chunk_size * sizeof(int));

    if (rank == 0) {
        array = (int*)malloc(ARRAY_SIZE * sizeof(int));
        for (int i = 0; i < ARRAY_SIZE; i++) array[i] = i + 1;
        printf("Process 0: Array initialized with values 1 to %d\n", ARRAY_SIZE);
    }

    // Distribute equal chunks to all processes
    MPI_Scatter(array, chunk_size, MPI_INT,
                local,  chunk_size, MPI_INT,
                0, MPI_COMM_WORLD);

    // Local sum
    int local_sum = 0;
    for (int i = 0; i < chunk_size; i++) local_sum += local[i];
    printf("Process %d: Local sum = %d\n", rank, local_sum);

    // Global reduction
    int global_sum = 0;
    MPI_Reduce(&local_sum, &global_sum, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("\nGlobal Sum  = %d\n", global_sum);
        printf("Expected    = 5050\n");
        printf("Average     = %.2f\n", (double)global_sum / ARRAY_SIZE);
        free(array);
    }

    free(local);
    MPI_Finalize();
    return 0;
}
