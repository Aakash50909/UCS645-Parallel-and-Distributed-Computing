# Assignment 6 — CUDA Programming
**Course:** UCS645 — Parallel and Distributed Computing  
**GPU:** NVIDIA RTX 3050 Laptop GPU (35W)

---

## Files

```
assignment6/
├── partA_device_query.cu
├── partB_array_sum.cu
├── partC_matrix_add.cu
├── Makefile
└── README.md
```

---

## Build & Run

```bash
make all
./partA
./partB
./partC
```

---

## Part A — Device Query

**Q1. What is the architecture and compute capability of your GPU?**

```c
cudaGetDeviceProperties(&prop, 0);
printf("Name: %s\n", prop.name);
printf("Compute Capability: %d.%d\n", prop.major, prop.minor);
```

Output:
```
Name: NVIDIA GeForce RTX 3050 Laptop GPU
Compute Capability: 8.6
```

The RTX 3050 Laptop GPU is based on the **Ampere** architecture with compute capability **8.6**.

---

**Q2. What are the maximum block dimensions for your GPU?**

```c
printf("Max Block Dim: %d x %d x %d\n", prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
printf("Max Threads Per Block: %d\n", prop.maxThreadsPerBlock);
```

Output:
```
Max Block Dim: 1024 x 1024 x 64
Max Threads Per Block: 1024
```

Maximum block dimensions are **1024 × 1024 × 64**, with a hard limit of **1024 threads per block** total.

---

**Q3. If the hardware's maximum grid dimension is 65535 and the maximum block dimension is 512 (1D), what is the maximum number of threads that can be launched?**

```c
printf("Max Grid Dim: %d x %d x %d\n", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
```

```
Max threads = 65535 × 512 = 33,553,920 threads
```

Each thread is identified by a unique `blockIdx.x * blockDim.x + threadIdx.x`, so the total is simply grid size multiplied by block size.

---

**Q4. Under what conditions might a programmer choose not to launch the maximum number of threads?**

- When the input data is smaller than the maximum thread count — extra threads would have nothing to process.
- When each thread uses a lot of registers or shared memory, launching fewer threads per block avoids resource exhaustion.
- When the problem is not parallelizable enough to benefit from that many threads.
- When launching too many threads adds overhead that outweighs the benefit for small workloads.

---

**Q5. What can limit a program from launching the maximum number of threads on a GPU?**

- **Registers:** Each thread uses registers. If a kernel uses many registers, fewer threads can run simultaneously on an SM.
- **Shared memory:** If a block requests more shared memory than available, the launch fails or stalls.
- **Global memory:** Not enough VRAM to hold input/output data.
- **Block size limit:** Hard cap of 1024 threads per block regardless of dimensions.
- **Grid size limit:** Hard cap on grid dimensions (65535 in y and z for older GPUs).

---

**Q6. What is shared memory? How much shared memory is on your GPU?**

```c
printf("Shared Memory Per Block: %zu bytes\n", prop.sharedMemPerBlock);
```

Output:
```
Shared Memory Per Block: 49152 bytes (48 KB)
```

Shared memory is a **fast, on-chip memory** shared by all threads within the same block. It is much faster than global memory and is used to avoid repeated slow reads from VRAM. The RTX 3050 Laptop has **48 KB of shared memory per block**.

---

**Q7. What is global memory? How much global memory is on your GPU?**

```c
printf("Total Global Memory: %.2f MB\n", prop.totalGlobalMem / (1024.0 * 1024.0));
```

Output:
```
Total Global Memory: 4096.00 MB (4 GB)
```

Global memory is the **main GPU VRAM** (GDDR6). It is accessible by all threads across all blocks. It is large but slow compared to shared memory — this is why coalesced memory access matters. The RTX 3050 Laptop has **4 GB** of global memory.

---

**Q8. What is constant memory? How much constant memory is on your GPU?**

```c
printf("Constant Memory: %zu bytes\n", prop.totalConstMem);
```

Output:
```
Constant Memory: 65536 bytes (64 KB)
```

Constant memory is a **read-only, cached memory** space. It is best for values that all threads read simultaneously (e.g., filter coefficients). When all threads in a warp access the same constant memory address, it is served from cache in a single transaction. All CUDA GPUs have **64 KB** of constant memory.

---

**Q9. What does warp size signify on a GPU? What is your GPU's warp size?**

```c
printf("Warp Size: %d\n", prop.warpSize);
```

Output:
```
Warp Size: 32
```

A **warp** is the smallest unit the GPU schedules for execution. All 32 threads in a warp run the same instruction at the same time. If threads in a warp take different paths (branch divergence), the GPU runs each path separately and masks out the others — reducing efficiency. Block sizes that are multiples of 32 avoid wasted threads in the last warp.

---

**Q10. Is double precision supported on your GPU?**

```c
printf("Double Precision Support: %s\n", prop.major >= 2 ? "Yes" : "No");
```

Output:
```
Double Precision Support: Yes
```

Yes, double precision (FP64) is supported. However, on the RTX 3050 Laptop (a consumer GPU), FP64 throughput is much lower than FP32 — roughly 1/32 the speed. For performance-critical work, single precision is preferred on this GPU.

---

## Part B — Array Sum

The kernel uses **parallel reduction** inside shared memory. Each block sums its chunk of the array using a tree reduction (stride halving), then writes one partial result to global memory. The host adds up the partial results.

**Steps performed:**
1. Allocate device memory with `cudaMalloc`
2. Copy host array to device with `cudaMemcpy`
3. Set block = 256 threads, grid = N / BLOCK_SIZE
4. Launch `sumKernel<<<grid, block>>>`
5. Copy partial results back to host
6. Add partial results on CPU to get final sum
7. Free device memory with `cudaFree`

**Output:**
```
Sum = 1024.00
```

All 1024 elements are `1.0f`, so the expected sum is 1024.

---

## Part C — Matrix Addition

The kernel assigns one thread per element. Each thread computes `C[row][col] = A[row][col] + B[row][col]` using its `blockIdx` and `threadIdx` to find its position in the matrix.

**Configuration:** 512×512 matrix, 16×16 thread blocks, 32×32 grid.

**Output:**
```
C[0][0] = 3
C[N-1][N-1] = 3
```

A is all 1s, B is all 2s, so every element of C is 3.

---

**Q1. How many floating-point operations are being performed in the matrix addition kernel?**

Each thread performs **1 addition**. With a 512×512 matrix:

```
512 × 512 = 262,144 additions
```

---

**Q2. How many global memory reads are being performed by your kernel?**

Each thread reads **2 values** (one from A, one from B):

```
2 × 512 × 512 = 524,288 reads
```

---

**Q3. How many global memory writes are being performed by your kernel?**

Each thread writes **1 value** (result into C):

```
1 × 512 × 512 = 262,144 writes
```

---

## Environment

| Component | Value |
|---|---|
| GPU | NVIDIA RTX 3050 Laptop GPU (35W) |
| Architecture | Ampere (sm_86) |
| VRAM | 4 GB GDDR6 |
| Warp Size | 32 |
| Shared Memory/Block | 48 KB |
| Constant Memory | 64 KB |
| Compute Capability | 8.6 |
