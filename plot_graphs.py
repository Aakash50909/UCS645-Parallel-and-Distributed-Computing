import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load data
df = pd.read_csv('results.csv')

# Unique questions
questions = df['Question'].unique()

plt.figure(figsize=(15, 5))

for i, q in enumerate(questions):
    subset = df[df['Question'] == q]
    threads = subset['Threads']
    times = subset['Time']
    
    # Calculate Speedup: T_serial / T_parallel
    t_serial = times.iloc[0] # Assumes first row is 1 thread
    speedup = t_serial / times
    
    # Theoretical Ideal Speedup (Linear)
    ideal = threads
    
    plt.subplot(1, 3, i+1)
    plt.plot(threads, speedup, 'o-', label='Actual Speedup', linewidth=2, markersize=8)
    plt.plot(threads, ideal, '--', label='Ideal (Linear)', color='gray', alpha=0.5)
    
    plt.title(f'{q} Speedup', fontsize=14)
    plt.xlabel('Number of Threads', fontsize=12)
    plt.ylabel('Speedup Factor', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.xticks([1, 2, 4, 8, 16])

plt.tight_layout()
plt.savefig('performance_graphs.png')
plt.show()
print("Graphs saved as performance_graphs.png")
