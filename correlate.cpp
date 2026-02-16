#include "correlate.h"
#include <cmath>
#include <vector>
#include <omp.h>
#include <iostream>

// --- Q1: Sequential Baseline ---
void correlate_sequential(int ny, int nx, const float* data, float* result) {
    for (int i = 0; i < ny; ++i) {
        for (int j = 0; j <= i; ++j) {
            double sum_x = 0, sum_y = 0, sum_xy = 0;
            double sum_x2 = 0, sum_y2 = 0;

            for (int k = 0; k < nx; ++k) {
                double x = data[i * nx + k];
                double y = data[j * nx + k];
                sum_x += x;
                sum_y += y;
                sum_xy += x * y;
                sum_x2 += x * x;
                sum_y2 += y * y;
            }

            double numer = nx * sum_xy - sum_x * sum_y;
            double denom = std::sqrt((nx * sum_x2 - sum_x * sum_x) * (nx * sum_y2 - sum_y * sum_y));
            result[i + j * ny] = (denom == 0) ? 0.0f : (float)(numer / denom);
        }
    }
}

// --- Q2/Q3: Parallel Optimized ---
void correlate(int ny, int nx, const float* data, float* result) {
    // 1. PRE-PROCESSING: Normalize rows to Mean=0, Variance=1
    // This simplifies the Correlation formula to just a Dot Product!
    std::vector<float> normalized_data(ny * nx);

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < ny; ++i) {
        double sum = 0.0, sq_sum = 0.0;
        for (int k = 0; k < nx; ++k) {
            float val = data[i * nx + k];
            sum += val;
            sq_sum += val * val;
        }
        
        double mean = sum / nx;
        double variance = sq_sum / nx - mean * mean;
        double inv_std_dev = (variance > 1e-9) ? 1.0 / std::sqrt(variance) : 0.0;
        
        for (int k = 0; k < nx; ++k) {
            normalized_data[i * nx + k] = (float)((data[i * nx + k] - mean) * inv_std_dev);
        }
    }

    // 2. PARALLEL DOT PRODUCT
    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < ny; ++i) {
        for (int j = 0; j <= i; ++j) {
            double dot_product = 0.0;
            
            // SIMD Vectorization for speed
            #pragma omp simd reduction(+:dot_product)
            for (int k = 0; k < nx; ++k) {
                dot_product += normalized_data[i * nx + k] * normalized_data[j * nx + k];
            }

            result[i + j * ny] = (float)(dot_product / nx);
        }
    }
}
