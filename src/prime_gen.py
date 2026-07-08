import cypari2
import numpy as np
import time
import os
import argparse
import json

# Initialize the PARI/GP engine inside Python
pari = cypari2.Pari()


def generate_ecpp_primes_for_ml(batch_size, min_digits, max_digits, seed_value):
    """
    Generates mathematically proven primes and tracks individual generation times.
    """
    rng = np.random.default_rng(seed_value)
    proven_primes = []

    print(f"Generating {batch_size} proven primes ({min_digits} to {max_digits} digits) with seed {seed_value}...\n")

    start_time = time.perf_counter()

    while len(proven_primes) < batch_size:
        current_digits = int(rng.integers(min_digits, max_digits + 1))

        lower_bound = 10 ** (current_digits - 1)
        upper_bound = 10 ** current_digits

        candidate = int(rng.integers(lower_bound, upper_bound))

        if pari.ispseudoprime(candidate):
            if pari.isprime(candidate, 3):
                proven_primes.append(candidate)

                print(
                    f"[{len(proven_primes)}/{batch_size}] ECPP Verified ({current_digits}-digit): {candidate}")

    generation_time = time.perf_counter() - start_time
    return proven_primes, generation_time


# --- 1. Generate your dataset ---
# CODE UPDATES:
# 1. Enabled terminal custom arguments pass in
# 2. Introduced random seeds as well as the ability to custom seeds via command lines
parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", type=int, default=100)
parser.add_argument("--min_digits", type=int, default=12)
parser.add_argument("--max_digits", type=int, default=15)
parser.add_argument("--seed", type=int, default=None)
args = parser.parse_args()

BATCH_SIZE = args.batch_size
MIN_DIGITS = args.min_digits
MAX_DIGITS = args.max_digits

SEED_VALUE = args.seed if args.seed is not None else int(time.time())

ml_dataset, gen_time = generate_ecpp_primes_for_ml(
    batch_size=BATCH_SIZE,
    min_digits=MIN_DIGITS,
    max_digits=MAX_DIGITS,
    seed_value=SEED_VALUE
)

# --- 2. Setup Data Directory ---
data_dir = os.path.join("..", "data")
raw_data_dir = os.path.join(data_dir, "raw")
os.makedirs(data_dir, exist_ok=True)
os.makedirs(raw_data_dir, exist_ok=True)

# --- 3. Save the Clean Primes File (For C Program) ---
base_filename = f"primes_{MIN_DIGITS}to{MAX_DIGITS}digits_seed{SEED_VALUE}"
primes_path = os.path.join(raw_data_dir, f"{base_filename}.txt")

with open(primes_path, "w") as file:
    for prime in ml_dataset:
        file.write(f"{prime}\n")

metadata = {
    "batch_size": BATCH_SIZE,
    "min_digits": MIN_DIGITS,
    "max_digits": MAX_DIGITS,
    "seed_used": SEED_VALUE,
    "total_generation_time_seconds": round(gen_time, 2),
    "average_time_per_prime_seconds": round(gen_time / BATCH_SIZE, 4)
}

metadata_path = os.path.join(raw_data_dir, f"{base_filename}_meta.json")

print(f"\nSuccessfully saved primes to: {primes_path}")
print(f"Successfully saved metadata to: {metadata_path}")