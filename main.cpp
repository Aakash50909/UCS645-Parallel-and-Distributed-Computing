#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <omp.h>
#include "correlate.h"

void fill_random(float* data, int n) {
    for (int i = 0; i < n; ++i) data[i] = (float)rand() / RAND_MAX;
}

int main(int argc, char* argv[]) {
    if (argc < 4) return 1;
    int ny = std::atoi(argv[1]);
    int nx = std::atoi(argv[2]);
    int mode = std::atoi(argv[3]); // 0=Seq, 1=Par

    std::vector<float> data(ny * nx);
    std::vector<float> result(ny * ny);
    fill_random(data.data(), ny * nx);

    double start = omp_get_wtime();
    if (mode == 0) correlate_sequential(ny, nx, data.data(), result.data());
    else correlate(ny, nx, data.data(), result.data());
    double end = omp_get_wtime();

    std::cout << "Time: " << (end - start) << "s" << std::endl;
    return 0;
}
