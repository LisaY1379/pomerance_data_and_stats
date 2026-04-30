import subprocess
import time
import os


def run_triple_benchmark(digits, seed):
    # 1. Define Paths
    input_file = os.path.join("..", "data", "raw", f"primes_{digits}digits_seed{seed}.txt")

    # CHANGED: Pure data is now saved as a .txt file
    out_pure = os.path.join("..", "data", "processed", f"triples_pure_{digits}digits_seed{seed}.txt")

    # Metrics data remains a .csv file
    out_metrics = os.path.join("..", "data", "processed", f"triples_metrics_{digits}digits_seed{seed}.csv")

    report_file = os.path.join("..", "reports", f"triple_benchmark_{digits}digits_seed{seed}.txt")

    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found. Run the prime generator first!")
        return

    print(f"Starting C Triple Generation for {digits} digits...")
    start_wall_time = time.perf_counter()

    # 2. Call the C Program with the THREE file arguments
    result = subprocess.run(
        ['./pomerance', input_file, out_pure, out_metrics],
        capture_output=True, text=True
    )

    end_wall_time = time.perf_counter()
    total_time = end_wall_time - start_wall_time

    # 3. Generate the Report
    with open(report_file, "w") as f:
        f.write("=== TRIPLE GENERATION BENCHMARK ===\n")
        f.write(f"Source Primes: {input_file}\n")
        f.write(f"Total Processing Time: {total_time:.4f} seconds\n")
        if total_time > 0:
            count = sum(1 for line in open(input_file))
            f.write(f"Average Search Time: {(total_time / count) * 1000:.4f} ms/triple\n")

    print(f"Done! Pure text data in: {out_pure}")
    print(f"Metrics CSV data in: {out_metrics}")
    print(f"Benchmark saved to: {report_file}")


run_triple_benchmark(digits=10, seed=42)