# Assignment 8 - GPU Accelerated Machine Learning

## What This Assignment Covers
- CUDA thread hierarchies and memory transfers
- Shared memory optimizations and parallel reductions
- ML primitive kernels: activations, loss functions, Adam optimizer
- Tiled matrix multiplication vs cuBLAS
- CNN layer benchmarking
- Full MNIST CNN training with PyTorch

---

## Folder Structure
```
assignment8/
├── ex01/
│   └── ex01_cuda_basics.cu       - GPU vs CPU speedup, bandwidth, launch configs
├── ex02/
│   └── ex02_memory_hierarchy.cu  - Shared memory, reductions, bank conflicts
├── ex03/
│   └── ex03_ml_primitives.cu     - Sigmoid, tanh, leaky ReLU, cross-entropy, Adam
├── ex04/
│   └── ex04_cnn_layers.cu        - Tiled GEMM, cuBLAS, MaxPool, BatchNorm
├── ex05/
│   └── ex05_mnist_cnn.py         - Full MNIST training with PyTorch
└── Makefile
```

---

## Requirements

### For ex01 to ex04 (CUDA C)
- NVIDIA GPU, Compute Capability 6.0+
- CUDA Toolkit 12.x
- nvcc compiler
- At least 4 GB VRAM

### For ex05 (Python/PyTorch)
- Python 3.10+
- PyTorch with CUDA support
- torchvision

Install Python deps:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

---

## How to Build (ex01-ex04)

```bash
make all
```

Or individually:
```bash
# ex01
nvcc -O2 -arch=sm_86 ex01/ex01_cuda_basics.cu -o ex01/ex01 -lm

# ex02
nvcc -O2 -arch=sm_86 ex02/ex02_memory_hierarchy.cu -o ex02/ex02 -lm

# ex03
nvcc -O2 -arch=sm_86 ex03/ex03_ml_primitives.cu -o ex03/ex03 -lm

# ex04 (needs cuBLAS)
nvcc -O2 -arch=sm_86 ex04/ex04_cnn_layers.cu -o ex04/ex04 -lcublas
```

Change `sm_86` to match your GPU. Common values:
- GTX 1060/1080: `sm_61`
- RTX 2080: `sm_75`
- RTX 3090: `sm_86`
- RTX 4090: `sm_89`

---

## How to Run

```bash
./ex01/ex01
./ex02/ex02
./ex03/ex03
./ex04/ex04
python ex05/ex05_mnist_cnn.py
```

---

## What Each Exercise Does

### EX01 - CUDA Basics
- Benchmarks memory bandwidth for 1, 8, 64, 256, 512 MB transfers (H2D and D2H)
- Compares CPU vs GPU time for vector sizes from 2^10 to 2^26
- Shows which thread block size (64/128/256/512/1024) is fastest
- Demonstrates the cost of warp divergence with a branch experiment

The **crossover point** (where GPU beats CPU) is usually around N=2^18 because smaller sizes are dominated by the time it takes to send data to the GPU over PCIe.

Multiples of 32 are preferred because CUDA runs threads in groups of 32 called warps. If your block is not a multiple of 32, the last warp has idle threads that still consume resources.

### EX02 - Memory Hierarchy
- Compares three sum reduction strategies: naive single-thread, shared memory tree, warp shuffle
- Shows how bank conflicts slow down shared memory at stride=32 (32 threads hit the same bank)
- Uses padding `tile[16][17]` to eliminate 2D bank conflicts
- Compares global vs shared memory histogram (atomicAdd contention)

Stride=1 is optimal because consecutive threads access consecutive banks — no conflicts. Stride=32 causes all 32 threads in a warp to land on the same bank, serializing the accesses.

### EX03 - ML Primitives
- Implements sigmoid, tanh, leaky ReLU, and ReLU backward as CUDA kernels
- Benchmarks all four at N=10^7 with bandwidth utilization
- Implements BCE loss with numerical clipping
- Implements cross-entropy using the log-sum-exp trick (numerically stable)
- Implements the CE gradient: softmax(logits) - one_hot(label)
- Implements fused Adam update in one kernel (no intermediate allocations)

### EX04 - CNN Layers
- Implements tiled matrix multiplication using shared memory tiles of size 16x16
- Benchmarks naive GPU, tiled GPU, and cuBLAS for matrix sizes 128 to 2048
- Reports GFLOPS for each
- Benchmarks MaxPool 2x2 and BatchNorm inference for typical MNIST feature maps [32,64,14,14]

Why tiled GEMM is slower than cuBLAS: cuBLAS uses Tensor Cores, FP16 computation, vectorized memory loads (128-bit), and highly tuned register blocking. Our implementation uses basic FP32 with simple tiling.

### EX05 - Full MNIST Training
Runs four training configurations and compares them:

| Config | Description |
|--------|-------------|
| Baseline | Adam, no BN, no Dropout |
| + BatchNorm | BatchNorm after each conv layer |
| + Dropout | Dropout(0.5) before final FC |
| SGD+Cosine | SGD with momentum and cosine LR schedule |

Also tests:
- Data augmentation: RandomRotation, RandomAffine, RandomErasing
- AMP (Automatic Mixed Precision) with `autocast` and `GradScaler`

Target: **97%+ test accuracy** on MNIST.

---

## Expected Output Samples

**EX01 Bandwidth:**
```
Size (MB)       H2D (MB/s)      D2H (MB/s)
1               XXXX.X          XXXX.X
512             XXXXX.X         XXXXX.X
```

**EX02 Reductions:**
```
Naive sum:         1048576  time: XXXXX.X us  correct: YES
Shared mem sum:    1048576  time: XXX.X us    BW: XX.X GB/s  correct: YES
Warp shuffle sum:  1048576  time: XXX.X us    BW: XX.X GB/s  correct: YES
```

**EX05 Training:**
```
Epoch   Train Loss     Train Acc%     Test Acc%      GPU Mem MB
1       0.2XXX         92.XX          97.XX          XXX.X
...
10      0.0XXX         99.XX          99.XX          XXX.X
```

---

## Notes
- Run in optimized mode (`-O2`) for accurate timing
- CUDA event timing is more accurate than CPU wall clock for GPU kernels
- For ex05, MNIST data will be auto-downloaded to `./data/` on first run
- Memory bandwidth numbers vary a lot by GPU — results above are approximate
