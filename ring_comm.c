// Exercise 1: Ring Communication
// Each process sends a message to the next in a ring.
// Process 0 starts with value 100, each process adds its rank.
// To compile: mpicc -o ring_comm ring_comm.c
// To run:     mpirun -np 4 ./ring_comm

#include <mpi.h>
#include <stdio.h>

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int next_rank = (rank + 1) % size;
    int prev_rank = (rank - 1 + size) % size;

    int value;

    if (rank == 0) {
        value = 100;
        printf("Process 0 starts with value: %d\n", value);
        // Send to next, then receive from last
        MPI_Send(&value, 1, MPI_INT, next_rank, 0, MPI_COMM_WORLD);
        MPI_Recv(&value, 1, MPI_INT, prev_rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        printf("Process 0 received final value back: %d\n", value);
    } else {
        // Receive from previous process
        MPI_Recv(&value, 1, MPI_INT, prev_rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        printf("Process %d received value: %d\n", rank, value);
        // Add own rank and forward
        value += rank;
        printf("Process %d sending value: %d to Process %d\n", rank, value, next_rank);
        MPI_Send(&value, 1, MPI_INT, next_rank, 0, MPI_COMM_WORLD);
    }

    MPI_Finalize();
    return 0;
}
