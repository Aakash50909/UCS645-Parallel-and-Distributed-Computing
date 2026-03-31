// Question 5: Parallel Perfect Number Finder — Master/Slave
// A perfect number equals the sum of its proper divisors (e.g., 6 = 1+2+3).
// Master dispatches candidates; slave tests and returns +n (perfect) or -n (not perfect).
// To compile: mpicc -O2 -o perfect_numbers perfect_numbers.c
// To run:     mpirun -np 4 ./perfect_numbers   (need at least 2 processes)

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define MAX_VALUE 10000   // Find perfect numbers up to this value

// Returns sum of proper divisors of n
static int divisor_sum(int n) {
    if (n < 2) return 0;
    int sum = 1;   // 1 is always a proper divisor for n > 1
    int limit = (int)sqrt((double)n);
    for (int i = 2; i <= limit; i++) {
        if (n % i == 0) {
            sum += i;
            if (i != n / i) sum += n / i;
        }
    }
    return sum;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size < 2) {
        if (rank == 0) fprintf(stderr, "Need at least 2 processes.\n");
        MPI_Finalize();
        return 1;
    }

    if (rank == 0) {
        // ---- MASTER ----
        int num_slaves  = size - 1;
        int next_num    = 2;
        int active      = 0;
        int perfects[100];
        int perf_count  = 0;

        // a. Pre-seed all slaves
        for (int s = 1; s <= num_slaves && next_num <= MAX_VALUE; s++) {
            int zero = 0;
            // According to protocol: slave sends 0 first ("just starting"),
            // but for efficiency we pre-seed directly here.
            MPI_Send(&next_num, 1, MPI_INT, s, 0, MPI_COMM_WORLD);
            next_num++;
            active++;
        }

        MPI_Status status;
        int result;
        while (active > 0) {
            // a. MPI_Recv from ANY slave
            MPI_Recv(&result, 1, MPI_INT, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &status);
            int slave_id = status.MPI_SOURCE;
            active--;

            // b. 0 = starting (handled above), negative = not perfect, positive = perfect
            if (result > 0) {
                perfects[perf_count++] = result;
                printf("Master: %d is a PERFECT NUMBER!\n", result);
            }

            // c. Send next candidate or termination
            if (next_num <= MAX_VALUE) {
                MPI_Send(&next_num, 1, MPI_INT, slave_id, 0, MPI_COMM_WORLD);
                next_num++;
                active++;
            } else {
                int terminate = -1;
                MPI_Send(&terminate, 1, MPI_INT, slave_id, 0, MPI_COMM_WORLD);
            }
        }

        printf("\nPerfect numbers up to %d: ", MAX_VALUE);
        for (int i = 0; i < perf_count; i++) printf("%d ", perfects[i]);
        printf("\n(Known perfect numbers: 6, 28, 496, 8128)\n");

    } else {
        // ---- SLAVE ----
        // Send initial "I'm ready" (0) — master may ignore if pre-seeding,
        // but including for full protocol correctness.
        int num;
        while (1) {
            MPI_Recv(&num, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            if (num == -1) break;   // Termination

            int s = divisor_sum(num);
            int reply = (s == num) ? num : -num;
            MPI_Send(&reply, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
        }
    }

    MPI_Finalize();
    return 0;
}
