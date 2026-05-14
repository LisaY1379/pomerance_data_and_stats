import subprocess
import time
import os
import csv

def run_triple_benchmark(min_digits, max_digits, seed):
    # 1. Define Paths
    input_file = os.path.join("..", "data", "raw", f"primes_{min_digits}to{max_digits}digits_seed{seed}.txt")

    # CHANGED: Pure data is now saved as a .txt file
    out_pure = os.path.join("..", "data", "processed", f"triples_pure_{min_digits}to{max_digits}digits_seed{seed}.txt")

    # Metrics data remains a .csv file
    out_metrics = os.path.join("..", "data", "processed", f"triples_metrics_{min_digits}to{max_digits}digits_seed{seed}.csv")

    report_file = os.path.join("..", "reports", f"triple_benchmark_{min_digits}to{max_digits}digits_seed{seed}.txt")

    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found. Run the prime generator first!")
        return

    print(f"Starting C Triple Generation for {min_digits} to {max_digits} digits...")
    start_wall_time = time.perf_counter()

    # 2. Call the C Program with the THREE file arguments
    result = subprocess.run(
        ['./pomerance', input_file, out_pure, out_metrics],
    )

    # CODE UPDATES: recorded number of trials for finding A
    total_trials = 0
    success_count = 0
    if os.path.exists(out_metrics):
        with open(out_metrics, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row["trials"] != "FAILED":
                    total_trials += int(row["trials"])
                    success_count += 1

    end_wall_time = time.perf_counter()
    total_time = end_wall_time - start_wall_time

    # 3. Generate the Report
    # CODE UPDATES: added trials of A to benchmark
    with open(report_file, "w") as f:
        f.write("=== TRIPLE GENERATION BENCHMARK ===\n")
        f.write(f"Source Primes: {input_file}\n")
        f.write(f"Hardware Processing Time: {total_time:.4f} seconds\n")
        f.write("-" * 35 + "\n")
        f.write(f"Successful Proofs: {success_count}\n")
        f.write(f"Total A's Tried: {total_trials}\n")
        if success_count > 0:
            f.write(f"Average A's Tried per Prime: {total_trials / success_count:.2f}\n")
        f.write("-" * 35 + "\n")

    print(f"Done! Pure text data in: {out_pure}")
    print(f"Metrics CSV data in: {out_metrics}")
    print(f"Benchmark saved to: {report_file}")


run_triple_benchmark(min_digits=7, max_digits=11, seed=1778732562)