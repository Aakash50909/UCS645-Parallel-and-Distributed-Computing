// Question 4: Parallel Prime Finder — Master/Slave with MPI_Send/MPI_Recv
// Master distributes numbers to slaves; slaves test and return +n (prime) or -n (not prime).
// To compile: mpicc -O2 -o primes primes.c
// To run:     mpirun -np 4 ./primes   (need at least 2 processes)

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define MAX_VALUE 1000   // Find all primes up to this value

// Simple primality test
static int is_prime(int n) {
    if (n < 2) return 0;
    if (n == 2) return 1;
    if (n % 2 == 0) return 0;
    int limit = (int)sqrt((double)n);
    for (int i = 3; i <= limit; i += 2)
        if (n % i == 0) return 0;
    return 1;
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
        int num_slaves = size - 1;
        int next_num   = 2;          // Next candidate to dispatch
        int active     = 0;          // How many slaves currently working
        int primes[MAX_VALUE + 1];
        int prime_count = 0;

        // Seed all slaves with their first number
        for (int s = 1; s <= num_slaves && next_num <= MAX_VALUE; s++) {
            MPI_Send(&next_num, 1, MPI_INT, s, 0, MPI_COMM_WORLD);
            next_num++;
            active++;
        }

        // Main dispatch loop
        MPI_Status status;
        int result;
        while (active > 0) {
            // a. Wait for a reply from ANY slave
            MPI_Recv(&result, 1, MPI_INT, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &status);
            int slave_id = status.MPI_SOURCE;
            active--;

            // b/c. Classify: zero = starting, positive = prime, negative = not prime
            if (result == 0) {
                // Slave just started (shouldn't happen here, handled above, but kept for robustness)
            } else if (result > 0) {
                primes[prime_count++] = result;
            }
            // negative result means not prime — we just don't store it

            // c. Send the next number if any remain
            if (next_num <= MAX_VALUE) {
                MPI_Send(&next_num, 1, MPI_INT, slave_id, 0, MPI_COMM_WORLD);
                next_num++;
                active++;
            } else {
                // Signal termination with -1
                int terminate = -1;
                MPI_Send(&terminate, 1, MPI_INT, slave_id, 0, MPI_COMM_WORLD);
            }
        }

        // Sort and print primes
        // Simple insertion sort (count is small)
        for (int i = 1; i < prime_count; i++) {
            int key = primes[i], j = i - 1;
            while (j >= 0 && primes[j] > key) { primes[j + 1] = primes[j]; j--; }
            primes[j + 1] = key;
        }

        printf("Primes up to %d (%d found):\n", MAX_VALUE, prime_count);
        for (int i = 0; i < prime_count; i++) {
            printf("%d ", primes[i]);
            if ((i + 1) % 20 == 0) printf("\n");
        }
        printf("\n");

    } else {
        // ---- SLAVE ----
        int num;
        // Send initial "I'm ready" signal (0) to master
        // Actually the master pre-seeds; but handle the generic protocol:
        // Wait for a number, test it, report, repeat.
        while (1) {
            MPI_Recv(&num, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            if (num == -1) break;   // Termination signal

            int reply = is_prime(num) ? num : -num;
            MPI_Send(&reply, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
        }
    }

    MPI_Finalize();
    return 0;
}
