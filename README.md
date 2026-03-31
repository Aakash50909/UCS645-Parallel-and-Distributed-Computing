# UCS645: Parallel & Distributed Computing

**Student:** Aakash Chandra — 102317242 | 3Q24
**System:** Pop!_OS (Linux) | AMD Ryzen 7 5800HS (16 Logical Processors)
**Compilers:** GCC/G++ (OpenMP + AVX2) · mpicc (MPICH)

---

## Repository Structure

This repository contains all lab assignments for UCS645, organized by branch. The first three assignments cover shared-memory parallelism with **OpenMP**; the final two cover distributed-memory parallelism with **MPI**.

| Branch | Assignment | Paradigm | Topic |
|---|---|---|---|
| `assignment1/openmp-matrix-mult` | Assignment 1 | OpenMP | Dense Matrix Multiplication — Naive vs Transposed |
| `assignment2/openmp-nbody-correlation` | Assignment 2 | OpenMP | N-Body Simulation + Parallel Matrix Correlation |
| `assignment3/openmp-advanced` | Assignment 3 | OpenMP | Advanced OpenMP (SIMD / synchronization) |
| `assignment4/mpi-intro` | Assignment 4 | MPI | Introduction to MPI — Point-to-Point & Collectives |
| `assignment5/mpi-advanced` | Assignment 5 | MPI | Blocking vs Non-Blocking · Master-Slave · Amdahl's Law |

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

*(See branch `lab3-submission' for more details)*

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

## Quick Start

### OpenMP assignments (1–3)

```bash
git checkout assignment1/openmp-matrix-mult   # or assignment2, assignment3
make
```

Compiler flags used: `-O3 -fopenmp -mavx2 -mfma`

### MPI assignments (4–5)

```bash
# Install MPICH if needed
sudo apt-get install mpich

git checkout assignment4/mpi-intro   # or assignment5/mpi-advanced
make

# Run any target (example)
mpirun -np 4 ./ring_comm
```

Each assignment directory contains its own `Makefile` with individual `run_*` targets and a `run_all` target.

---

## Technology Stack

| Tool | Purpose |
|---|---|
| GCC/G++ with `-fopenmp` | Shared-memory parallelism (Assignments 1–3) |
| `-O3 -mavx2 -mfma` | Auto-vectorization + SIMD FMA instructions |
| mpicc (MPICH) | Distributed-memory parallelism (Assignments 4–5) |
| `omp_get_wtime()` / `MPI_Wtime()` | High-resolution wall-clock timing |
| Amdahl's Law | Theoretical speedup ceiling and parallel fraction analysis |
