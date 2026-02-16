import subprocess
import matplotlib.pyplot as plt
import re

# Configuration
rows = 2000
cols = 2000
threads_list = [1, 2, 4, 8, 16]
times = []
executable = "./correlate_app"

print(f"--- Benchmarking Matrix Correlation ({rows}x{cols}) ---")

# 1. Run Sequential Baseline (Mode 0)
print("Running Sequential Baseline...", end="", flush=True)
result = subprocess.run([executable, str(rows), str(cols), "0"], capture_output=True, text=True)
seq_time_match = re.search(r"Time: ([0-9.]+)s", result.stdout)
seq_time = float(seq_time_match.group(1)) if seq_time_match else 0.0
print(f" Done! ({seq_time:.4f}s)")

# 2. Run Parallel Scaling (Mode 1)
for t in threads_list:
    print(f"Running Parallel with {t} threads...", end="", flush=True)
    
    # Set OMP_NUM_THREADS via environment variable for this subprocess
    env = {"OMP_NUM_THREADS": str(t)}
    
    result = subprocess.run([executable, str(rows), str(cols), "1"], capture_output=True, text=True, env=env)
    time_match = re.search(r"Time: ([0-9.]+)s", result.stdout)
    
    if time_match:
        times.append(float(time_match.group(1)))
        print(f" Done! ({times[-1]:.4f}s)")
    else:
        times.append(0.0)
        print(" Failed!")

# 3. Calculate Speedup
speedups = [seq_time / t if t > 0 else 0 for t in times]
ideal_speedup = threads_list

# 4. Plotting
plt.figure(figsize=(10, 6))

# Plot Actual Speedup
plt.plot(threads_list, speedups, 'o-', label='Actual Speedup', color='blue', linewidth=2, markersize=8)

# Plot Ideal Speedup (Dotted Line)
plt.plot(threads_list, ideal_speedup, '--', label='Ideal Linear Speedup', color='gray', alpha=0.5)

plt.title(f'Performance Scaling: Matrix Correlation ({rows}x{cols})', fontsize=14)
plt.xlabel('Number of Threads', fontsize=12)
plt.ylabel('Speedup Factor (vs Sequential)', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()
plt.xticks(threads_list)

plt.savefig('correlation_performance.png')
print("\nGraph saved as 'correlation_performance.png'")
