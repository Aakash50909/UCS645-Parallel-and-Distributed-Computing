# Assignment 7 - CUDA Part II

## What This Assignment Covers
- Launching CUDA kernels and making threads do different things
- Allocating memory and moving it between CPU and GPU
- Understanding CUDA thread block layouts
- Querying GPU device properties
- Measuring and comparing theoretical vs actual memory bandwidth

---

## Files
```
assignment7/
├── problem1.cu    - Sum of first N integers (two threads, two methods)
├── problem2.cu    - Merge sort: pipelined CPU vs CUDA GPU
├── problem3.cu    - Vector addition with timing and bandwidth measurement
└── Makefile
```

---

## How to Build

Make sure you have the CUDA toolkit installed and `nvcc` is available.

```bash
make all
```

Or build one at a time:

```bash
nvcc -O2 problem1.cu -o problem1
nvcc -O2 problem2.cu -o problem2
nvcc -O2 problem3.cu -o problem3
```

---

## How to Run

```bash
./problem1
./problem2
./problem3
```

---

## Problem Descriptions

### Problem 1 - Two Threads Doing Different Work
Two CUDA threads run at the same time. Thread 0 adds 1 to 1024 the slow way (loop). Thread 1 uses the formula n*(n+1)/2 to get the answer instantly. Both should print 524800.

### Problem 2 - Merge Sort Comparison
- **Part a**: CPU merge sort using a pipelining approach (bottom-up iterative)
- **Part b**: CUDA GPU merge sort where different thread blocks handle different merge pairs in parallel
- **Part c**: Both print their timing so you can compare. For N=1000 the CPU is likely faster due to small size and GPU transfer overhead — that is expected and worth noting.

### Problem 3 - Vector Addition with Profiling
This one uses statically declared device memory (no `cudaMalloc` needed). It:
- Runs the vector add kernel
- Times the kernel using CUDA events
- Reads the GPU's memory clock and bus width to compute **theoretical bandwidth**
- Computes **measured bandwidth** from bytes read/written divided by time
- Prints both so you can see the gap

The gap between theoretical and measured bandwidth exists because of memory access patterns, caching, and overhead. This is normal.

---

## Notes on Static Device Memory (Problem 3)
Using `__device__` creates memory directly on the GPU without `cudaMalloc`. You cannot pass a `__device__` symbol as a kernel argument — instead the kernel just accesses it by name directly. Use `cudaMemcpyToSymbol` and `cudaMemcpyFromSymbol` to move data in and out.

---

## Expected Output (approximate)

**Problem 1:**
```
Sum using iterative method: 524800
Sum using formula method:   524800
```

**Problem 2:**
```
Pipelining CPU merge sort time: X.XXX ms
CUDA GPU merge sort time:       X.XXX ms
Speedup (CPU/GPU): X.XXx
Both sorts correct: YES
```

**Problem 3:**
```
Kernel execution time: X.XXXX ms
Theoretical bandwidth: XXX.XX GB/s
Measured bandwidth:    XX.XX GB/s
Bandwidth efficiency:  XX.X%
Result correct: YES
```

---

## Requirements
- NVIDIA GPU with CUDA support
- CUDA Toolkit (nvcc)
- Linux or WSL (recommended)
- Build in Release/optimized mode: `-O2` flag is already in the Makefile
