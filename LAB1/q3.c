#include <stdio.h>
#include <omp.h>

static long num_steps = 100000;
double step;

int main() {
    int i;
    double x, pi, sum = 0.0;
    double start, end;

    step = 1.0 / (double)num_steps;

    start = omp_get_wtime();

    #pragma omp parallel for private(x) reduction(+:sum)
    for (i = 0; i < num_steps; i++) {
        x = (i + 0.5) * step;
        sum = sum + 4.0 / (1.0 + x * x);
    }

    pi = step * sum;
    end = omp_get_wtime();

    printf("Pi: %f\n", pi);
    printf("Time: %f seconds\n", end - start);

    return 0;
}
