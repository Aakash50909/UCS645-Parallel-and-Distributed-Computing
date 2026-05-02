# UCS645: Parallel & Distributed Computing

**Student:** Aakash Chandra — 102317242 | 3Q24
**System:** Pop!_OS (Linux) | AMD Ryzen 7 5800HS (16 Logical Processors) | NVIDIA RTX 3050 Laptop GPU (35W)
**Compilers:** GCC/G++ (OpenMP + AVX2) · mpicc (MPICH) · nvcc (CUDA 12.x)

---

## Repository Structure

This repository contains all lab assignments for UCS645, organized by branch. The first three assignments cover shared-memory parallelism with **OpenMP**, the next two cover distributed-memory parallelism with **MPI**, and the final three cover **GPU programming with CUDA**.

| Branch | Assignment | Paradigm | Topic |
|---|---|---|---|
| `assignment1/openmp-matrix-mult` | Assignment 1 | OpenMP | Dense Matrix Multiplication — Naive vs Transposed |
| `assignment2/openmp-nbody-correlation` | Assignment 2 | OpenMP | N-Body Simulation + Parallel Matrix Correlation |
| `assignment3/openmp-advanced` | Assignment 3 | OpenMP | Advanced OpenMP (SIMD / synchronization) |
| `assignment4/mpi-intro` | Assignment 4 | MPI | Introduction to MPI — Point-to-Point & Collectives |
| `assignment5/mpi-advanced` | Assignment 5 | MPI | Blocking vs Non-Blocking · Master-Slave · Amdahl's Law |
| `assignment6/cuda-intro` | Assignment 6 | CUDA | Device Query · Array Sum · Matrix Addition |
| `assignment7/cuda-kernels` | Assignment 7 | CUDA | Thread Tasks · Merge Sort · Vector Add + Bandwidth |
| `assignment8/cuda-ml` | Assignment 8 | CUDA | GPU-Accelerated ML — Reductions · Activations · CNN |

---

## Assignment Summaries

### Assignment 1 — OpenMP: Dense Matrix Multiplication

Analyzes three implementations of N×N matrix multiplication (N=1000):

- **Sequential** — single-threaded baseline
- **Basic OpenMP** — `#pragma omp parallel for` on the outer loop
- **Optimized OpenMP** — matrix transposition for spatial locality, compiled with `-O3 -mavx2`

**Key result:** The naive parallel version was *slower* than sequential (0.48x) due to memory bandwidth contention. The transposed + optimized version achieved **14.98x speedup** on 16 threads (93.6% efficiency), with an estimated parallel fraction of **P ≈ 99.5%** via Amdahl's Law.

---

### Assignment 2 — OpenMP: N-Body Simulation & Matrix Correlation

Two problems demonstrating different parallelization strategies:

**Molecular Dynamics (N-Body):** Computes Lennard-Jones potential forces between N particles in 3D (O(N²)). Uses `#pragma omp atomic` to handle symmetric force updates (Fᵢⱼ = −Fⱼᵢ) and `schedule(dynamic)` for load balancing. Near-linear speedup at low thread counts; atomic contention becomes the bottleneck at 16 threads.

**Parallel Matrix Correlation:** Computes the full m×m Pearson correlation matrix for m=2000 vectors. Pre-normalizes each vector (mean=0, variance=1) so that correlation reduces to a simple dot product — enabling full AVX2/FMA vectorization of the inner loop. Near-linear scaling to 8 threads; memory bandwidth becomes the limiting factor at 16 threads.

---

### Assignment 3 — OpenMP: Advanced Topics

*(See branch `lab3-submission` for more details)*

---

### Assignment 4 — MPI: Introduction

Four exercises covering the fundamentals of MPI communication:

- **Ring Communication** — point-to-point message passing in a ring topology
- **Parallel Array Sum** — `MPI_Scatter` + `MPI_Reduce` (expected result: 5050)
- **Global Max/Min** — `MPI_MAXLOC` / `MPI_MINLOC` with source-rank tracking
- **Parallel Dot Product** — vectors A=[1..8], B=[8..1], expected result: 120

---

### Assignment 5 — MPI: Advanced Communication Patterns

Five questions covering non-blocking I/O, collective operations, and master-slave parallelism:

- **Q1 — DAXPY** (`X[i] = a·X[i] + Y[i]`, N=2¹⁶): MPI scatter/gather with sequential baseline for speedup measurement
- **Q2 — Broadcast Race**: Linear `MyBcast` O(p) vs `MPI_Bcast` O(log p) — empirical timing across 2/4/8/16 processes
- **Q3 — Distributed Dot Product**: 500M-element vectors, each process generates its own chunk; `MPI_Bcast` + `MPI_Reduce`; Amdahl's Law analysis table
- **Q4 — Parallel Prime Finder**: Master-slave with `MPI_ANY_SOURCE`; `+n` = prime, `-n` = not prime protocol
- **Q5 — Perfect Number Finder**: Same master-slave pattern; tests divisor sums up to √n; verifiable against 6, 28, 496, 8128

---

### Assignment 6 — CUDA: Introduction

Three parts introducing GPU programming fundamentals on the RTX 3050 Laptop (sm_86, 4 GB VRAM):

- **Part A — Device Query**: Queries and explains 10 GPU properties — compute capability (8.6 / Ampere), max block dimensions (1024×1024×64), warp size (32), shared memory (48 KB/block), global memory (4 GB), constant memory (64 KB), and double precision support
- **Part B — Array Sum**: Parallel reduction using shared memory tree (stride halving); 256 threads/block; partial sums merged on host
- **Part C — Matrix Addition**: One thread per element; 512×512 matrix with 16×16 blocks; analyzes FLOP count (262K adds), global reads (524K), and global writes (262K)

---

### Assignment 7 — CUDA: Kernels & Profiling

Three problems building practical CUDA programming skills:

- **Problem 1 — Multi-task threads**: Two threads run simultaneously — Thread 0 computes the sum of 1..1024 iteratively; Thread 1 uses the direct formula. Both produce 524800.
- **Problem 2 — Merge Sort**: Pipelined bottom-up CPU merge sort vs CUDA parallel merge sort (N=1000). Different thread blocks handle separate merge pairs in parallel. Includes timing comparison — PCIe overhead typically makes CPU faster at this size, which is expected and discussed.
- **Problem 3 — Vector Add + Bandwidth**: Uses static `__device__` globals (no `cudaMalloc`). Measures kernel execution time via CUDA events, computes **theoretical bandwidth** from `memoryClockRate × memoryBusWidth × 2`, computes **measured bandwidth** from bytes transferred divided by time, and prints both for comparison.

---

### Assignment 8 — CUDA: GPU-Accelerated Machine Learning

Five exercises building from low-level CUDA to a full CNN training pipeline:

- **EX01 — CUDA Basics**: Bandwidth benchmark (1–512 MB H2D/D2H), CPU vs GPU speedup table for N=2¹⁰ to 2²⁶ with crossover analysis, thread block size sweep (64/128/256/512/1024), warp divergence timing experiment
- **EX02 — Memory Hierarchy**: Three reduction strategies (naive single-thread / shared memory tree / warp shuffle with `__shfl_down_sync`), bank conflict profiling at strides 1–32, `tile[16][17]` padding fix, shared-memory histogram vs global atomicAdd
- **EX03 — ML Primitives**: Forward kernels for sigmoid, tanh, leaky ReLU, ReLU backward; numerically stable cross-entropy (log-sum-exp trick); CE gradient kernel (`softmax − one_hot`); fused Adam optimizer (single kernel, no intermediate allocations). All verified against PyTorch with atol ≤ 1e-4.
- **EX04 — Tiled GEMM & CNN Layers**: Tiled matrix multiply (TILE=16) vs naive GPU vs cuBLAS across sizes 128–2048, GFLOPS reported; MaxPool 2×2 and BatchNorm inference benchmarked on [32, 64, 14, 14] tensors; roofline analysis
- **EX05 — Full MNIST CNN**: PyTorch training pipeline with 4-config ablation study (Baseline / +BatchNorm / +Dropout / SGD+CosineAnnealingLR), data augmentation (RandomRotation + RandomAffine + RandomErasing), and AMP training with `autocast` + `GradScaler`. Target: ≥97% test accuracy.

---

## Quick Start

### OpenMP assignments (1–3)

```bash
git checkout assignment1/openmp-matrix-mult   # or assignment2, assignment3
make
```

Compiler flags used: `-O3 -fopenmp -mavx2 -mfma`

### MPI assignments (4–5)

```bash
sudo apt-get install mpich

git checkout assignment4/mpi-intro   # or assignment5/mpi-advanced
make

mpirun -np 4 ./ring_comm
```

Each assignment directory contains its own `Makefile` with individual `run_*` targets and a `run_all` target.

### CUDA assignments (6–8)

```bash
# Check your GPU's compute capability first
nvidia-smi

git checkout assignment6/cuda-intro   # or assignment7, assignment8
make
```

Compile flags used: `-O2 -arch=sm_86` (change `sm_86` to match your GPU — see table below)

| GPU | arch flag |
|---|---|
| GTX 1060 / 1080 | `sm_61` |
| RTX 2080 | `sm_75` |
| RTX 3050 / 3090 | `sm_86` |
| RTX 4090 | `sm_89` |

For Assignment 8 EX05 (Python/PyTorch):

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
python ex05/ex05_mnist_cnn.py
```

---

## Technology Stack

| Tool | Purpose |
|---|---|
| GCC/G++ with `-fopenmp` | Shared-memory parallelism (Assignments 1–3) |
| `-O3 -mavx2 -mfma` | Auto-vectorization + SIMD FMA instructions |
| mpicc (MPICH) | Distributed-memory parallelism (Assignments 4–5) |
| nvcc (CUDA Toolkit 12.x) | GPU kernel compilation (Assignments 6–8) |
| cuBLAS | Optimized GPU matrix multiplication (Assignment 8 EX04) |
| PyTorch + torchvision | CNN training and reference validation (Assignment 8 EX05) |
| `omp_get_wtime()` / `MPI_Wtime()` | High-resolution wall-clock timing (OpenMP / MPI) |
| CUDA Events (`cudaEventRecord`) | High-resolution GPU kernel timing (CUDA) |
| Amdahl's Law | Theoretical speedup ceiling and parallel fraction analysis |
