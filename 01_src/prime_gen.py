import cypari2
import numpy as np
import time
import os
import subprocess

# Initialize the PARI/GP engine inside Python
pari = cypari2.Pari()


def generate_ecpp_primes_for_ml(batch_size, digits, seed_value):
    """
    Generates mathematically proven primes and tracks individual generation times.
    """
    rng = np.random.default_rng(seed_value)
    proven_primes = []
    generation_times = []  # Store every individual time

    print(f"Generating {batch_size} proven primes ({digits}-digit) with seed {seed_value}...\n")

    lower_bound = 10 ** (digits - 1)
    upper_bound = 10 ** digits

    current_prime_start = time.perf_counter()

    while len(proven_primes) < batch_size:
        candidate = int(rng.integers(lower_bound, upper_bound))

        if pari.ispseudoprime(candidate):
            if pari.isprime(candidate, 3):
                proven_primes.append(candidate)

                # Record the time for this specific prime
                time_taken = time.perf_counter() - current_prime_start
                generation_times.append(time_taken)

                print(
                    f"[{len(proven_primes)}/{batch_size}] ECPP Verified: {candidate} | Time: {time_taken * 1000:.4f} ms")

                # Reset stopwatch
                current_prime_start = time.perf_counter()

    # Return BOTH lists so we can save them later
    return proven_primes, generation_times


# --- 1. Generate your dataset ---
BATCH_SIZE = 100
DIGITS = 10
SEED_VALUE = 42

start_time = time.time()
ml_dataset, times_list = generate_ecpp_primes_for_ml(batch_size=BATCH_SIZE, digits=DIGITS, seed_value=SEED_VALUE)

# Calculate summary statistics
total_gen_time = sum(times_list)
average_time = total_gen_time / len(times_list)
overall_script_time = time.time() - start_time

# --- 2. Setup Data Directory ---
data_dir = os.path.join("..", "data")
reports_dir = os.path.join("..", "reports")
os.makedirs(data_dir, exist_ok=True)
os.makedirs(reports_dir, exist_ok=True)

# --- 3. Save the Clean Primes File (For C Program) ---
primes_filename = f"primes_{DIGITS}digits_seed{SEED_VALUE}.txt"
primes_path = os.path.join(data_dir, primes_filename)

with open(primes_path, "w") as file:
    for prime in ml_dataset:
        file.write(f"{prime}\n")

# --- 4. Save the Benchmark Report (For You) ---
benchmark_filename = f"benchmark_{DIGITS}digits_seed{SEED_VALUE}.txt"
benchmark_path = os.path.join(reports_dir, benchmark_filename)

with open(benchmark_path, "w") as file:
    # Write the summary header
    file.write("=== GENERATION BENCHMARK REPORT ===\n")
    file.write(f"Digits: {DIGITS}\n")
    file.write(f"Seed Used: {SEED_VALUE}\n")
    file.write(f"Total Primes Generated: {BATCH_SIZE}\n")
    file.write("-" * 35 + "\n")
    file.write(f"Total Prime Generation Time: {total_gen_time * 1000:.4f} ms\n")
    file.write(f"Average Time Per Prime:      {average_time * 1000:.4f} ms\n")
    file.write(f"Total Script Runtime:        {overall_script_time:.4f} seconds\n")
    file.write("===================================\n\n")

    # Write the individual times
    file.write("--- Individual Generation Times ---\n")
    for i, (prime, t) in enumerate(zip(ml_dataset, times_list)):
        file.write(f"Prime {i + 1} ({prime}): {t * 1000:.4f} ms\n")

print(f"\nSuccessfully saved primes to: {primes_path}")
print(f"Successfully saved benchmark report to: {benchmark_path}")