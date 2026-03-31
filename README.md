# UCS645
# Assignment 4: Introduction to MPI

**Student:** Aakash Chandra — 102317242 3Q24
**Course:** UCS645: Parallel & Distributed Computing
**System:** Pop!_OS (Linux) | **Compiler:** mpicc (MPICH/OpenMPI)

---

## Project Overview

This assignment introduces the Message Passing Interface (MPI) through four programming exercises covering point-to-point communication, collective operations, and performance measurement. Each exercise is implemented as a standalone C program compiled via a shared Makefile.

---

## File Structure

```
assignment4/
├── Makefile
├── ring_comm.c        # Exercise 1: Ring Communication
├── array_sum.c        # Exercise 2: Parallel Array Sum
├── max_min.c          # Exercise 3: Global Max and Min
└── dot_product.c      # Exercise 4: Parallel Dot Product
```

---

## How to Build & Run

### Prerequisites

```bash
sudo apt-get install mpich
```

### Build all targets

```bash
make
```

### Run individual exercises

```bash
mpirun -np 4 ./ring_comm
mpirun -np 4 ./array_sum
mpirun -np 4 ./max_min
mpirun -np 4 ./dot_product
```

### Run all at once

```bash
make run_all
```

---

## Exercise Descriptions

### Exercise 1: Ring Communication (`ring_comm.c`)

Each process sends a message to the next process in a ring topology. Process 0 initializes value = 100; each subsequent process adds its rank before forwarding. The value wraps back to Process 0 after traversing all processes.

**Key concept:** `(rank + 1) % size` for ring wrapping. Uses alternating `MPI_Send`/`MPI_Recv` to avoid deadlock.

**Expected output (4 processes):**
```
Process 0 starts with value: 100
Process 1 received value: 100, sending: 101
Process 2 received value: 101, sending: 103
Process 3 received value: 103, sending: 106
Process 0 received final value back: 106
```

---

### Exercise 2: Parallel Array Sum (`array_sum.c`)

An array of integers 1–100 is distributed across processes using `MPI_Scatter`. Each process computes the sum of its local chunk. `MPI_Reduce` with `MPI_SUM` combines the partial sums. The global result is verified against the expected value of 5050, and the average (50.5) is also printed.

**Key concepts:** `MPI_Scatter`, `MPI_Reduce`, load distribution.

---

### Exercise 3: Global Max and Min (`max_min.c`)

Each process independently generates 10 random integers (0–1000). Local max and min are found, then `MPI_Reduce` with `MPI_MAXLOC` and `MPI_MINLOC` returns both the global extreme values **and** the rank that produced them.

**Key concept:** `MPI_2INT` datatype with `{value, rank}` struct for location-aware reductions.

---

### Exercise 4: Parallel Dot Product (`dot_product.c`)

Vectors A = [1,2,3,4,5,6,7,8] and B = [8,7,6,5,4,3,2,1] are scattered across processes. Each process computes its partial dot product. `MPI_Reduce` sums all partial products on Rank 0.

**Expected result:** 1×8 + 2×7 + 3×6 + 4×5 + 5×4 + 6×3 + 7×2 + 8×1 = **120**

---

## Performance Metrics

| Metric | Formula |
|---|---|
| Speedup (Sp) | T₁ / Tₚ |
| Efficiency (Ep) | Sp / p |
| Communication overhead | Tcomm / Ttotal |

Run each program with 1, 2, 4, and 8 processes and record `MPI_Wtime()` output to populate the table.

---

## Key MPI Functions Used

| Function | Purpose |
|---|---|
| `MPI_Send` / `MPI_Recv` | Point-to-point blocking communication |
| `MPI_Scatter` | Distribute array from root to all |
| `MPI_Gather` | Collect array from all to root |
| `MPI_Reduce` | Combine values with an operation (SUM, MAX, MIN) |
| `MPI_Bcast` | Broadcast from root to all |
| `MPI_Wtime()` | High-resolution wall-clock timer |
