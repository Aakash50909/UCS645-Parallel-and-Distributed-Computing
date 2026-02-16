#!/bin/bash

# Compile the codes first (ensuring optimization is ON)
echo "Compiling..."
g++ -O3 -fopenmp molecules.cpp -o q1
g++ -O3 -fopenmp smith_waterman.cpp -o q2
g++ -O3 -fopenmp heat_diffusion.cpp -o q3

# Output file
OUTPUT_FILE="results.csv"
echo "Question,Threads,Size,Time" > $OUTPUT_FILE

# --- Configuration (INCREASED SIZES for better graphs) ---
# Q1: Particles. 15000 particles will make O(N^2) heavy enough.
Q1_SIZE=15000 
# Q2: Sequence Length. 15000x15000 matrix is heavy.
Q2_SIZE=15000
# Q3: Grid Size. 4000x4000 grid is memory intensive.
Q3_SIZE=4000

# Function to run benchmark
run_test() {
    EXE=$1
    Q_NAME=$2
    SIZE=$3
    THREADS=$4
    
    echo "Running $Q_NAME with $THREADS threads..."
    
    # Run and extract time using grep/awk (assumes your C++ output format)
    # capturing the output line like "Time: 0.264398s"
    RESULT=$($EXE $SIZE $THREADS)
    TIME=$(echo $RESULT | grep -oP 'Time: \K[0-9.]+')
    
    echo "$Q_NAME,$THREADS,$SIZE,$TIME" >> $OUTPUT_FILE
}

# --- Execution Loop ---
for THREADS in 1 2 4 8 16; do
    run_test "./q1" "Q1_MD" $Q1_SIZE $THREADS
    run_test "./q2" "Q2_DNA" $Q2_SIZE $THREADS
    run_test "./q3" "Q3_Heat" $Q3_SIZE $THREADS
done

echo "Done! Results saved to $OUTPUT_FILE"
cat $OUTPUT_FILE
