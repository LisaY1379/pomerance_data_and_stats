import subprocess
import time
import os
import csv
from collections import defaultdict

def run_incremental_benchmark(digits, target_triples=20):
    data_dir = os.path.join("..", "data", "final_data")
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    old_pure_file = os.path.join(data_dir, f"prime_{digits}_digits.csv")
    stateful_input_file = os.path.join(data_dir, f"stateful_prime_{digits}_digits.txt")
    out_pure_new = os.path.join(data_dir, f"prime_{digits}_digits_NEWONLY.txt")
    out_metrics_new = os.path.join(data_dir, f"prime_{digits}_digits_metrics_NEWONLY.csv")
    final_20_file = os.path.join(data_dir, f"prime_{digits}_digits_FINAL_20.csv")
    report_file = os.path.join("..", "reports", f"triple_benchmark_{digits}_digits_incremental.txt")

    print(f"Step 1: Reading existing triples from {old_pure_file} to build context state...")
    existing_data = defaultdict(list)

    if os.path.exists(old_pure_file):
        with open(old_pure_file, "r") as f:
            reader = csv.reader(f)
            for row in reader:
                if not row or "FAILED" in row[0].upper() or "PRIME" in row[0].upper():
                    continue
                if len(row) >= 3:
                    p, a, x0 = row[0].strip(), row[1].strip(), row[2].strip()
                    if p.isdigit() and a.isdigit() and x0.isdigit():
                        existing_data[p].append((a, x0))
    else:
        print(f"Warning: Original file {old_pure_file} not found. Starting fresh.")

    print(f"Loaded existing data for {len(existing_data)} primes.")

    primes_to_process = 0
    with open(stateful_input_file, "w") as f:
        for p, pairs in existing_data.items():
            if len(pairs) >= target_triples:
                continue

            primes_to_process += 1
            line_parts = [p, str(len(pairs))]
            for a, x0 in pairs:
                line_parts.extend([a, x0])

            f.write(" ".join(line_parts) + "\n")

    if primes_to_process == 0:
        print("All primes already have the target number of triples! Exiting.")
        return

    print(f"Step 2: Starting C Generation for {primes_to_process} primes to reach {target_triples} triples...")
    start_wall_time = time.perf_counter()

    subprocess.run(
        ['./pomerance', stateful_input_file, out_pure_new, out_metrics_new, str(target_triples)]
    )

    end_wall_time = time.perf_counter()
    total_time = end_wall_time - start_wall_time
    print(f"C generation finished in {total_time:.2f} seconds.")

    print(f"Step 3: Safely merging and SORTING 4-column data into {final_20_file}...")

    all_rows = []
    total_trials = 0
    success_count = 0

    if os.path.exists(old_pure_file):
        with open(old_pure_file, "r") as fold:
            reader = csv.reader(fold)
            for row in reader:
                if not row or "FAILED" in row[0].upper() or "PRIME" in row[0].upper():
                    continue
                if len(row) >= 4:
                    all_rows.append([row[0].strip(), row[1].strip(), row[2].strip(), row[3].strip()])

    if os.path.exists(out_metrics_new):
        with open(out_metrics_new, "r") as fnew:
            reader = csv.reader(fnew)
            for row in reader:
                if not row or "FAILED" in row[0].upper() or "PRIME" in row[0].upper():
                    continue
                if len(row) >= 4:
                    all_rows.append([row[0].strip(), row[1].strip(), row[2].strip(), row[3].strip()])
                    total_trials += int(row[3].strip())
                    success_count += 1

    all_rows.sort(key=lambda x: int(x[0]))

    with open(final_20_file, "w", newline='') as fout:
        writer = csv.writer(fout)
        writer.writerow(['prime', 'A', 'x0', 'trials'])
        writer.writerows(all_rows)

    report_dir = os.path.dirname(report_file)
    if not os.path.exists(report_dir):
        os.makedirs(report_dir)

    with open(report_file, "w") as f:
        f.write("=== INCREMENTAL GENERATION BENCHMARK ===\n")
        f.write(f"Targeting total: {target_triples} triples per prime\n")
        f.write(f"Digits parameter: {digits}\n")
        f.write(f"Hardware Processing Time: {total_time:.4f} seconds\n")
        f.write("-" * 35 + "\n")
        f.write(f"New Successful Proofs Generated: {success_count}\n")
        f.write(f"Total New A's Tried: {total_trials}\n")
        if success_count > 0:
            f.write(f"Average New A's Tried per Proof: {total_trials / success_count:.2f}\n")
        f.write("-" * 35 + "\n")

    print(f"Done! Merged and SORTED data (with trials) saved to: {final_20_file}")
    print(f"Benchmark report saved to: {report_file}")

run_incremental_benchmark(digits=9, target_triples=20)