import subprocess
import time
import os
import csv


def run_triple_benchmark(min_digits, max_digits, seed):
    input_file = os.path.join("..", "data", "final_data",
                              f"triples_pure_{min_digits}to{max_digits}digits_seed{seed}.txt")

    out_pure = os.path.join("..", "data", "final_data", f"triples_pure_{min_digits}to{max_digits}digits_seed{seed}_new.txt")

    out_metrics = os.path.join("..", "data", "final_data",
                               f"triples_metrics_{min_digits}to{max_digits}digits_seed{seed}_new.csv")

    report_file = os.path.join("..", "reports", f"triple_benchmark_{min_digits}to{max_digits}digits_seed{seed}.txt")

    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found. Run the prime generator first!")
        return

    print("Inspecting input data for FAILED records...")
    with open(input_file, "r") as f:
        lines = f.readlines()

    valid_lines = [line for line in lines if "FAILED" not in line]

    if len(valid_lines) < len(lines):
        removed_count = len(lines) - len(valid_lines)
        print(f"Scrubbed {removed_count} FAILED records. Rewriting clean input file...")
        with open(input_file, "w") as f:
            f.writelines(valid_lines)
    else:
        print("Input data is clean. No FAILED records found.")

    print(f"Starting C Triple Generation for {min_digits} to {max_digits} digits...")
    start_wall_time = time.perf_counter()

    result = subprocess.run(
        ['./pomerance', input_file, out_pure, out_metrics],
    )

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

run_triple_benchmark(min_digits=12, max_digits=15, seed=1778731660)