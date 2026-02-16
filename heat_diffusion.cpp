#include <iostream>
#include <vector>
#include <omp.h>
#include <cmath>

const int STEPS = 100;

void heat_diffusion(int N, int threads) {
    std::vector<std::vector<double>> T(N, std::vector<double>(N, 0.0));
    std::vector<std::vector<double>> T_new(N, std::vector<double>(N, 0.0));
    T[N/2][N/2] = 1000.0;

    omp_set_num_threads(threads);
    double start_time = omp_get_wtime();

    for (int t = 0; t < STEPS; ++t) {
        #pragma omp parallel for collapse(2) schedule(static)
        for (int i = 1; i < N - 1; ++i) {
            for (int j = 1; j < N - 1; ++j) {
                T_new[i][j] = 0.25 * (T[i+1][j] + T[i-1][j] + T[i][j+1] + T[i][j-1]);
            }
        }
        #pragma omp parallel for collapse(2)
        for (int i = 1; i < N - 1; ++i) {
            for (int j = 1; j < N - 1; ++j) {
                T[i][j] = T_new[i][j];
            }
        }
    }
    double total_heat = 0.0;
    #pragma omp parallel for collapse(2) reduction(+:total_heat)
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            total_heat += T[i][j];
        }
    }
    double end_time = omp_get_wtime();
    std::cout << "Threads: " << threads << " | Grid: " << N << "x" << N 
              << " | Time: " << (end_time - start_time) << "s" << std::endl;
}

int main(int argc, char* argv[]) {
    int size = (argc > 1) ? atoi(argv[1]) : 1000;
    int threads = (argc > 2) ? atoi(argv[2]) : 4;
    heat_diffusion(size, threads);
    return 0;
}
