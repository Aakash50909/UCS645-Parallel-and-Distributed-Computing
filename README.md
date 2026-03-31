# UCS645
# Assignment 5: MPI Part II — Blocking vs Non-Blocking Communication

**Student:** Aakash Chandra — 102317242 3Q24
**Course:** UCS645: Parallel & Distributed Computing
**System:** Pop!_OS (Linux) | **Compiler:** mpicc (MPICH/OpenMPI)

---

## Project Overview

This assignment explores advanced MPI patterns: blocking vs non-blocking communication (deadlock trap and overlap fix), collective operations (`MPI_Bcast`, `MPI_Reduce`), and master-slave parallelism. Five questions are implemented, each demonstrating a distinct HPC design principle.

---

## File Structure

```
assignment5/
├── Makefile
├── daxpy.c              # Q1: DAXPY loop — MPI speedup measurement
├── bcast_race.c         # Q2: Broadcast Race — Linear vs Tree (MPI_Bcast)
├── dot_amdahl.c         # Q3: Distributed Dot Product & Amdahl's Law
├── primes.c             # Q4: Parallel Prime Finder (master-slave)
└── perfect_numbers.c    # Q5: Parallel Perfect Number Finder (master-slave)
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

### Run individual questions

```bash
mpirun -np 4 ./daxpy
mpirun -np 4 ./bcast_race
mpirun -np 4 ./dot_amdahl
mpirun -np 4 ./primes
mpirun -np 4 ./perfect_numbers
```

### Run scaling experiments (Q2 and Q3 recommended)

```bash
make run_bcast_race      # Runs with 2, 4, 8 processes automatically
make run_dot_amdahl      # Runs with 1, 2, 4, 8 processes automatically
```

---

## Question Descriptions

### Q1: DAXPY (`daxpy.c`)

**Operation:** `X[i] = a * X[i] + Y[i]` over vectors of size 2¹⁶ = 65,536 elements.

The sequential baseline runs entirely on Rank 0, timed with `MPI_Wtime()`. The parallel version uses `MPI_Scatter` to distribute chunks, each process computes its local DAXPY, then `MPI_Gather` collects results. Speedup and efficiency are printed at the end.

**Key takeaway:** Even for this embarrassingly parallel operation, communication overhead (`MPI_Scatter`/`MPI_Gather` latency) limits ideal linear speedup — visible via the efficiency metric.

---

### Q2: Broadcast Race (`bcast_race.c`)

**Objective:** Empirically demonstrate the algorithmic gap between a naive linear broadcast and `MPI_Bcast`.

| Method | Algorithm | Time Complexity |
|---|---|---|
| `MyBcast` (for-loop) | Rank 0 sends sequentially to 1, 2, 3… | O(p) |
| `MPI_Bcast` (library) | Tree-based — 0→1, then {0,1}→{2,3}, etc. | O(log p) |

**Expected observation:** As process count grows, `MyBcast` time grows linearly while `MPI_Bcast` time grows logarithmically. The gap is most visible at 8 and 16 processes.

**Sample results table to fill:**

| Processes | MyBcast (s) | MPI_Bcast (s) | Speedup |
|---|---|---|---|
| 2 | | | |
| 4 | | | |
| 8 | | | |
| 16 | | | |

---

### Q3: Distributed Dot Product & Amdahl's Law (`dot_amdahl.c`)

**Vectors:** 500 million elements each. Each process generates its own local chunk (memory-efficient — no full vector on any single process).

**Workflow:**
1. Rank 0 broadcasts a `multiplier` via `MPI_Bcast`
2. Each process generates: `A[i] = 1.0`, `B[i] = 2.0 * multiplier`
3. Local dot product computed in a `for` loop
4. `MPI_Reduce` with `MPI_SUM` aggregates on Rank 0

**Speedup & Efficiency Table (fill after runs):**

| Processes (p) | Time (Tp) | Speedup (T1/Tp) | Efficiency (Sp/p) |
|---|---|---|---|
| 1 | | 1.00 | 100% |
| 2 | | | |
| 4 | | | |
| 8 | | | |

**Amdahl's Law analysis:** Perfect linear speedup is not achieved because:
- The `MPI_Reduce` collective is a serial synchronization point
- Memory bandwidth is the bottleneck for large arrays (dot product has low arithmetic intensity)
- MPI communication latency and startup costs add fixed overhead

---

### Q4: Parallel Prime Finder (`primes.c`)

**Pattern:** Master-slave with dynamic work distribution.

**Master (Rank 0):**
- Loops from 2 to `MAX_VALUE` (default: 1000)
- Issues `MPI_Recv` from `MPI_ANY_SOURCE`; interprets reply sign
- Sends next candidate to the just-freed slave via `MPI_Send`
- Terminates slaves with signal `-1`

**Slave (Rank ≥ 1):**
- Receives a number, tests primality (trial division up to √n)
- Returns `+n` if prime, `-n` if not
- Loops until termination signal

**Known primes up to 1000:** 168 primes, ending with 997.

---

### Q5: Parallel Perfect Number Finder (`perfect_numbers.c`)

**Definition:** A perfect number equals the sum of its proper divisors. E.g., 6 = 1+2+3.

**Pattern:** Same master-slave dynamic dispatch as Q4.

**Master:** Dispatches candidates 2..`MAX_VALUE` (default: 10000). Interprets `+n` = perfect, `-n` = not perfect.

**Slave:** Computes divisor sum using trial division up to √n. Returns `+n` or `-n`.

**Known perfect numbers ≤ 10000:** 6, 28, 496, 8128.

**Note:** Perfect numbers are extremely rare. Beyond 8128, the next is 33,550,336 — so results within MAX_VALUE=10000 are deterministic and verifiable.

---

## Core Concepts Demonstrated

| Concept | Where Used |
|---|---|
| Hiding communication latency | Q1 (overlap pattern from Assignment theory) |
| Algorithmic complexity (O(p) vs O(log p)) | Q2 |
| Amdahl's Law — serial bottlenecks | Q3 |
| Dynamic work distribution (master-slave) | Q4, Q5 |
| `MPI_ANY_SOURCE` for load balancing | Q4, Q5 |
| Non-blocking: `MPI_Isend`/`MPI_Irecv` | Theory examples in assignment PDF |
