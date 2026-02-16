#include <iostream>
#include <vector>
#include <cmath>
#include <omp.h>
#include <cstdlib>

struct double3 { double x, y, z; };

void compute_forces(int N, std::vector<double3>& pos, std::vector<double3>& forces, double& total_energy) {
    double epsilon = 1.0, sigma = 1.0;
    std::fill(forces.begin(), forces.end(), double3{0, 0, 0});
    total_energy = 0.0;

    #pragma omp parallel 
    {
        double local_energy = 0.0;
        #pragma omp for schedule(dynamic)
        for (int i = 0; i < N; ++i) {
            for (int j = i + 1; j < N; ++j) {
                double dx = pos[i].x - pos[j].x;
                double dy = pos[i].y - pos[j].y;
                double dz = pos[i].z - pos[j].z;
                double r2 = dx*dx + dy*dy + dz*dz;
                
                if (r2 > 0.001) {
                    double inv_r2 = 1.0 / r2;
                    double inv_r6 = inv_r2 * inv_r2 * inv_r2;
                    double force_scalar = 24.0 * epsilon * inv_r2 * (2.0 * inv_r6 * inv_r6 - inv_r6); 
                    local_energy += 4.0 * epsilon * (inv_r6 * inv_r6 - inv_r6);

                    #pragma omp atomic
                    forces[i].x += force_scalar * dx;
                    #pragma omp atomic
                    forces[i].y += force_scalar * dy;
                    #pragma omp atomic
                    forces[i].z += force_scalar * dz;

                    #pragma omp atomic
                    forces[j].x -= force_scalar * dx;
                    #pragma omp atomic
                    forces[j].y -= force_scalar * dy;
                    #pragma omp atomic
                    forces[j].z -= force_scalar * dz;
                }
            }
        }
        #pragma omp atomic
        total_energy += local_energy;
    }
}

int main(int argc, char* argv[]) {
    int N = (argc > 1) ? atoi(argv[1]) : 5000;
    int threads = (argc > 2) ? atoi(argv[2]) : 4;
    omp_set_num_threads(threads);
    std::vector<double3> pos(N);
    std::vector<double3> forces(N);
    double total_energy = 0;
    for(int i=0; i<N; i++) pos[i] = {drand48()*100, drand48()*100, drand48()*100};

    double start_time = omp_get_wtime();
    compute_forces(N, pos, forces, total_energy);
    double end_time = omp_get_wtime();

    std::cout << "Threads: " << threads << " | Particles: " << N 
              << " | Time: " << (end_time - start_time) << "s" << std::endl;
    return 0;
}
