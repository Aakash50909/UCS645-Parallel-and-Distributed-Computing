#include <iostream>
#include <vector>
#include <algorithm>
#include <omp.h>
#include <string>

void smith_waterman(const std::string& seqA, const std::string& seqB, int threads) {
    int rows = seqA.length() + 1;
    int cols = seqB.length() + 1;
    std::vector<int> H(rows * cols, 0);

    omp_set_num_threads(threads);
    double start_time = omp_get_wtime();

    for (int k = 2; k < rows + cols - 1; ++k) {
        int i_start = std::max(1, k - cols + 1);
        int i_end = std::min(rows - 1, k - 1);

        #pragma omp parallel for schedule(static)
        for (int i = i_start; i <= i_end; ++i) {
            int j = k - i; 
            int match = (seqA[i-1] == seqB[j-1]) ? 3 : -3;
            int score_diag = H[(i-1)*cols + (j-1)] + match;
            int score_up   = H[(i-1)*cols + j] - 2;   
            int score_left = H[i*cols + (j-1)] - 2;
            H[i*cols + j] = std::max(0, std::max(score_diag, std::max(score_up, score_left)));
        }
    }

    double end_time = omp_get_wtime();
    std::cout << "Threads: " << threads << " | Matrix: " << rows << "x" << cols 
              << " | Time: " << (end_time - start_time) << "s" << std::endl;
}

int main(int argc, char* argv[]) {
    int size = (argc > 1) ? atoi(argv[1]) : 5000;
    int threads = (argc > 2) ? atoi(argv[2]) : 4;
    std::string seqA(size, 'A');
    std::string seqB(size, 'T');
    smith_waterman(seqA, seqB, threads);
    return 0;
}
