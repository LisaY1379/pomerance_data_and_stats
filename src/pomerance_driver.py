import os
import sys
import csv
import time
import subprocess
import pandas as pd
import numpy as np
from core.difficulty_parameter_a import process_prime_data

try:
    import cypari2

    pari = cypari2.Pari()
except ImportError:
    print("❌ Critical Error: 'cypari2' library not detected in this Python environment.")
    sys.exit(1)


def compile_c_core():
    """
    Locates and automatically compiles the C language core binary.
    """
    print("Step 0: Checking and compiling C core executable...")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    c_source_file = os.path.join(script_dir, "pomerance.c")
    executable_path = os.path.join(script_dir, "pomerance")

    if not os.path.exists(c_source_file):
        print(f"❌ Error: Source file not found at {c_source_file}")
        if os.path.exists(executable_path):
            print(f"⚠️ Utilizing existing binary at {executable_path}")
            return executable_path
        sys.exit(1)

    compile_strategies = [
        (['gcc-15', '-O3', '-fopenmp', c_source_file, '-o', executable_path, '-lm'],
         "GCC 15 with OpenMP (Max Performance)"),
        (['gcc', '-O3', '-fopenmp', c_source_file, '-o', executable_path, '-lm'],
         "Standard GCC with OpenMP (Multi-core)"),
        (['gcc', '-O3', c_source_file, '-o', executable_path, '-lm'],
         "Standard GCC (Single-thread Fallback)")
    ]

    for cmd, desc in compile_strategies:
        print(f"  Attempting: {desc}...")
        try:
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                print(f"  🔥 SUCCESS: Compiled successfully using {desc}!")
                return executable_path
        except FileNotFoundError:
            pass

    print("\n❌ Final Compilation FAILED! All compilation strategies exhausted.")
    sys.exit(1)


def generate_proven_primes(number_of_primes, digits, seed_value):
    """
    Generates mathematically proven ECPP primes natively via cypari2 engine.
    """
    print(f"Step 1: Generating {number_of_primes} fresh proven primes ({digits} digits) with seed {seed_value}...")
    rng = np.random.default_rng(seed_value)
    proven_primes = []

    lower_bound = 10 ** (digits - 1)
    upper_bound = 10 ** digits

    while len(proven_primes) < number_of_primes:
        candidate = int(rng.integers(lower_bound, upper_bound))
        if pari.ispseudoprime(candidate):
            if pari.isprime(candidate, 3):
                proven_primes.append(candidate)
                print(f"  [{len(proven_primes)}/{number_of_primes}] ECPP Verified: {candidate}")

    return proven_primes


def run_incremental_benchmark(digits=9, triple_amount_for_each_prime=20, number_of_primes=20, seed=None):
    """
    Universal one-click generation entry point with clear explicit terminology.
    """
    # 0. Compile C Engine
    executable_path = compile_c_core()
    print("-" * 65)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, ".."))
    data_dir = os.path.join(project_root, "data")

    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    seed_value = seed if seed is not None else int(time.time())

    # 📁 Setup temporary pipeline scratch files
    stateful_input_file = os.path.join(data_dir, f"tmp_stateful_primes_{digits}d.txt")
    out_pure_new = os.path.join(data_dir, f"tmp_pure_{digits}d_NEWONLY.txt")
    out_metrics_new = os.path.join(data_dir, f"tmp_metrics_{digits}d_NEWONLY.csv")
    tmp_merged_4col = os.path.join(data_dir, f"tmp_merged_4col_{digits}d.csv")
    tmp_with_a_5col = os.path.join(data_dir, f"tmp_with_a_5col_{digits}d.csv")

    # 🛠️ AUTO-INCREMENT VERSION CONFLICT RESOLUTION (Starting at v2)
    version = 2
    while True:
        candidate_filename = f"{digits}_digits_v{version}.csv"
        final_output_path = os.path.join(data_dir, candidate_filename)
        if not os.path.exists(final_output_path):
            break
        version += 1

    # --- Step 1: Fresh Prime Generation ---
    fresh_primes = generate_proven_primes(number_of_primes, digits, seed_value)

    # --- Step 2: Construct Fresh Entrypoints for C ---
    print(f"\nStep 2: Preparing data boundaries for C Core allocation matrix...")
    with open(stateful_input_file, "w") as f:
        for p in fresh_primes:
            f.write(f"{p} 0\n")

    # --- Step 3: Call C Pseudo-Prime Triple Verification Engine ---
    print(f"\nStep 3: Launching C Core execution loop targeting {triple_amount_for_each_prime} curves per prime...")
    subprocess.run(
        [executable_path, stateful_input_file, out_pure_new, out_metrics_new, str(triple_amount_for_each_prime)])

    # Parse and sort 4-column raw metrics data stream
    all_rows = []
    if os.path.exists(out_metrics_new):
        with open(out_metrics_new, "r") as fnew:
            reader = csv.reader(fnew)
            for row in reader:
                if not row or "FAILED" in row[0].upper() or "PRIME" in row[0].upper():
                    continue
                if len(row) >= 4:
                    all_rows.append([row[0].strip(), row[1].strip(), row[2].strip(), row[3].strip()])

    all_rows.sort(key=lambda x: int(x[0]))

    with open(tmp_merged_4col, "w", newline='') as fout:
        writer = csv.writer(fout)
        writer.writerow(['prime', 'A', 'x0', 'trials'])
        writer.writerows(all_rows)

    # --- Step 4: Parameter A Integration (Calling the Core module directly) ---
    print("\nStep 4: Redirecting data to core module for Parameter A compute...")
    process_prime_data(input_file=tmp_merged_4col, output_file=tmp_with_a_5col)

    # --- Step 5: Bridge Cross-Runtime call to SageMath for Parameter B ---
    print("\nStep 5: Invoking SageMath via background subprocess for Parameter B...")
    sage_script_path = os.path.join(script_dir, "core", "difficulty_parameter_b.sage")
    if not os.path.exists(sage_script_path):
        sage_script_path = os.path.join(script_dir, "difficulty_parameter_b.sage")

    sage_cmd = ["sage", sage_script_path, tmp_with_a_5col, final_output_path]

    try:
        subprocess.run(sage_cmd, capture_output=True, text=True, check=True)
        print("  [+] SageMath Group Action Class Number calculations completed.")
    except FileNotFoundError:
        print("\n❌ Environment Execution Error: 'sage' binary utility not detected on system PATH.")
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        print(f"\n❌ SageMath Subprocess Crash! Output details:\n{e.stderr}")
        sys.exit(1)

    # --- Step 6: Spotless Garbage Clean up Engine ---
    print("\nStep 6: Purging all temporary scratch files and pure TXT outputs...")
    garbage_list = [stateful_input_file, out_pure_new, out_metrics_new, tmp_merged_4col, tmp_with_a_5col]
    for trash in garbage_list:
        if os.path.exists(trash):
            os.remove(trash)

    print(f"\n==================================================================")
    print(f"🎉 FRESH END-TO-END PIPELINE SHUTDOWN SUCCESSFUL!")
    print(f"👉 New Clean Asset Generated -> data/{os.path.basename(final_output_path)}")
    print(f"==================================================================\n")


if __name__ == "__main__":
    run_incremental_benchmark(digits=9, triple_amount_for_each_prime=20, number_of_primes=100)