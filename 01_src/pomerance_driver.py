import subprocess
import time
import os
import csv


def run_triple_benchmark(min_digits, max_digits, seed, num_triples=10):
    input_file = os.path.join("..", "data", "final_data",
                              f"triples_pure_{min_digits}to{max_digits}digits_seed{seed}.txt")

    clean_primes_file = os.path.join("..", "data", "final_data",
                                     f"primes_only_{min_digits}to{max_digits}digits_seed{seed}.txt")

    out_pure = os.path.join("..", "data", "final_data",
                            f"triples_pure_{min_digits}to{max_digits}digits_seed{seed}_new.txt")
    out_metrics = os.path.join("..", "data", "final_data",
                               f"triples_metrics_{min_digits}to{max_digits}digits_seed{seed}_new.csv")
    report_file = os.path.join("..", "reports", f"triple_benchmark_{min_digits}to{max_digits}digits_seed{seed}.txt")

    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found.")
        return

    print("Inspecting input data, extracting unique primes...")
    unique_primes = set()

    with open(input_file, "r") as f:
        for line in f:
            if "FAILED" in line or not line.strip():
                continue
            parts = line.split(',')
            if len(parts) > 0:
                p_str = parts[0].strip()
                if p_str.isdigit():
                    unique_primes.add(int(p_str))

    sorted_primes = sorted(list(unique_primes))
    with open(clean_primes_file, "w") as f:
        for p in sorted_primes:
            f.write(f"{p}\n")

    print(f"Found {len(sorted_primes)} unique valid primes. Starting C Generation...")
    start_wall_time = time.perf_counter()

    result = subprocess.run(
        ['./pomerance', clean_primes_file, out_pure, out_metrics, str(num_triples)]
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
        f.write(f"Source Primes extracted from: {input_file}\n")
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

run_triple_benchmark(min_digits=12, max_digits=15, seed=1778731660, num_triples=5)