# Parallel & Distributed Computing - Lab Assignment 2
**Student:** Aakash Chandra -102317242 3Q24
**Course:** UCS645: Parallel & Distributed Computing  
**System:** Pop!_OS (Linux) | **Compiler:** G++ (OpenMP enabled)

---

## 📊 Performance Analysis Overview

This repository contains parallel implementations of three distinct computational problems using **OpenMP**. The goal was to analyze the speedup, efficiency, and scalability of different parallelization strategies on a multi-core CPU.

![Performance Graphs](performance_graphs.png)
*(Figure 1: Speedup vs. Number of Threads for all three problems)*

---

## 1. Molecular Dynamics (N-Body Simulation)
**Problem:** Calculate Lennard-Jones potential forces and total energy for `N` particles in 3D space.  
**Complexity:** $O(N^2)$

### Implementation Strategy
* **Parallelization:** The outer loop over particles was parallelized using `#pragma omp parallel for`.
* **Race Condition Handling:** Since $F_{ij} = -F_{ji}$, updates to the `forces` array are symmetric. I used `#pragma omp atomic` to safely update force vectors without locking the entire array, minimizing synchronization overhead.
* **Load Balancing:** `schedule(dynamic)` was used to handle potential load imbalances if particle density varies.

### Performance Discussion
* **Observation:** The algorithm shows near-linear speedup for lower thread counts (1-4 threads) because the problem is **compute-bound**. The heavy floating-point calculations ($r^{-6}, r^{-12}$) dominate execution time.
* **Bottleneck:** As thread count increases (8-16), efficiency drops slightly due to **atomic contention**. Multiple threads attempting to update the force on the same particle simultaneously
