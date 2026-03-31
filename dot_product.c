// Exercise 4: Parallel Dot Product
// A = [1,2,3,4,5,6,7,8], B = [8,7,6,5,4,3,2,1]
// Expected dot product = 1*8 + 2*7 + ... + 8*1 = 120
// To compile: mpicc -o dot_product dot_product.c
// To run:     mpirun -np 4 ./dot_product

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#define VECTOR_SIZE 8

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int chunk_size = VECTOR_SIZE / size;
    int* a_full = NULL;
    int* b_full = NULL;
    int* a_local = (int*)malloc(chunk_size * sizeof(int));
    int* b_local = (int*)malloc(chunk_size * sizeof(int));

    if (rank == 0) {
        a_full = (int*)malloc(VECTOR_SIZE * sizeof(int));
        b_full = (int*)malloc(VECTOR_SIZE * sizeof(int));
        for (int i = 0; i < VECTOR_SIZE; i++) {
            a_full[i] = i + 1;
            b_full[i] = VECTOR_SIZE - i;
        }
        printf("Vector A: ");
        for (int i = 0; i < VECTOR_SIZE; i++) printf("%d ", a_full[i]);
        printf("\nVector B: ");
        for (int i = 0; i < VECTOR_SIZE; i++) printf("%d ", b_full[i]);
        printf("\n\n");
    }

    MPI_Scatter(a_full, chunk_size, MPI_INT, a_local, chunk_size, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Scatter(b_full, chunk_size, MPI_INT, b_local, chunk_size, MPI_INT, 0, MPI_COMM_WORLD);

    int local_dot = 0;
    for (int i = 0; i < chunk_size; i++) local_dot += a_local[i] * b_local[i];
    printf("Process %d: partial dot product = %d\n", rank, local_dot);

    int global_dot = 0;
    MPI_Reduce(&local_dot, &global_dot, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("\nGlobal Dot Product = %d\n", global_dot);
        printf("Expected           = 120\n");
        free(a_full);
        free(b_full);
    }

    free(a_local);
    free(b_local);
    MPI_Finalize();
    return 0;
}
