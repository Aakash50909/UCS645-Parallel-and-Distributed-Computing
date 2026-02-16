#!/bin/bash
make clean && make
ROWS=1000
COLS=1000

echo "--- 1. SEQUENTIAL BASELINE ---"
perf stat -e cycles,instructions,cache-misses ./correlate_app $ROWS $COLS 0

echo "--- 2. PARALLEL SCALING ---"
for THREADS in 1 2 4 8 16; do
    echo "Running with $THREADS threads..."
    export OMP_NUM_THREADS=$THREADS
    ./correlate_app $ROWS $COLS 1
done
