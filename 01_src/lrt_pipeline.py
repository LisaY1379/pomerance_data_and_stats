import os
import csv
import subprocess
import tempfile
import time
import cypari2
import numpy as np
import pandas as pd
import scipy.stats as stats

# Initialize the PARI/GP engine inside Python
pari = cypari2.Pari()


def generate_ecpp_primes_for_interval(start_val, end_val, required_count, rng, history_file="used_primes_history.txt"):
    """
    Attempts to sample `required_count` unique primes within a specific physical interval [start_val, end_val].
    Features a Global Blacklist to prevent cross-run collisions and a circuit breaker for prime deserts.
    """
    global_history = set()

    # 1. Load historical blacklist to ensure absolute uniqueness across multiple script executions
    if os.path.exists(history_file):
        with open(history_file, 'r', encoding='utf-8') as f:
            for line in f:
                val = line.strip()
                if val:
                    global_history.add(int(val))

    proven_primes = set()
    attempts = 0
    # Safety lock: Max 2000 blind attempts per interval
    max_attempts = max(2000, required_count * 100)

    while len(proven_primes) < required_count and attempts < max_attempts:
        attempts += 1
        # Blind sample within the current 1.1x micro-window
        candidate = int(rng.integers(start_val, end_val + 1))

        # Check against both the current batch AND the global history
        if candidate not in proven_primes and candidate not in global_history:
            if pari.ispseudoprime(candidate):
                if pari.isprime(candidate, 3):
                    proven_primes.add(candidate)

    # 2. Append newly found primes to the global blacklist
    if proven_primes:
        with open(history_file, 'a', encoding='utf-8') as f:
            for p in proven_primes:
                f.write(f"{p}\n")

    return list(proven_primes)


# ================= LRT Engine (Maintained exact math core) =================

def geometric_log_likelihood(n, y_bar, p):
    if p <= 0 or p >= 1: return -np.inf
    return n * np.log(p) + n * (y_bar - 1) * np.log(1 - p)


def run_pava_decreasing(y_bars):
    p_hat = [1.0 / y for y in y_bars]
    w = [1.0] * len(y_bars)
    m = list(p_hat)
    while True:
        violates = False
        for i in range(len(m) - 1):
            if m[i] < m[i + 1] - 1e-9:
                new_val = (m[i] * w[i] + m[i + 1] * w[i + 1]) / (w[i] + w[i + 1])
                m[i] = m[i + 1] = new_val
                new_w = w[i] + w[i + 1]
                w[i] = w[i + 1] = new_w
                violates = True
        if not violates: break
    return m


def get_level_probabilities(k):
    dp = np.zeros((k + 1, k + 1))
    dp[1][1] = 1.0
    for i in range(2, k + 1):
        for l in range(1, i + 1):
            dp[i][l] = ((i - 1) / i) * dp[i - 1][l] + (1 / i) * dp[i - 1][l - 1]
    return dp[k][1:]


def perform_lrt(n, y_bars):
    k = len(y_bars)
    p_hat = [1.0 / y for y in y_bars]
    p_tilde = run_pava_decreasing(y_bars)
    l_unconstrained = sum(geometric_log_likelihood(n, y_bars[i], p_hat[i]) for i in range(k))
    l_constrained = sum(geometric_log_likelihood(n, y_bars[i], p_tilde[i]) for i in range(k))
    T_stat = 2 * (l_unconstrained - l_constrained)
    if T_stat < 1e-8: T_stat = 0.0

    if T_stat == 0:
        p_value = 1.0
    else:
        p_value = 0.0
        P_lk = get_level_probabilities(k)
        for l_minus_1, prob in enumerate(P_lk):
            df = k - (l_minus_1 + 1)
            if df > 0: p_value += prob * stats.chi2.sf(T_stat, df)
    return T_stat, p_value, p_hat, p_tilde


# ================= Main Stratified Pipeline =================

def run_pipeline(target_digits=3, triplets_per_interval=1, num_simulations=10, seed_value=None):
    if seed_value is None:
        seed_value = int(time.time())
    rng = np.random.default_rng(seed_value)

    print(f"\n🚀 Starting {target_digits}-digit Decimal Radar Sweep (1.1x Micro-Slicing Mode)...")

    # ================= 1. Path Resolution =================
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)

    c_exec_path = os.path.join(script_dir, "pomerance")
    if os.name == 'nt' and not os.path.exists(c_exec_path): c_exec_path += ".exe"
    if not os.path.exists(c_exec_path): raise FileNotFoundError(f"❌ C executable not found at: {c_exec_path}")

    # Create distinct directories based on target_digits
    lrt_data_dir = os.path.join(project_root, 'data', 'lrt_data', f'{target_digits}digits_Pipeline')
    sim_data_dir = os.path.join(lrt_data_dir, "Simulation_Data")
    os.makedirs(sim_data_dir, exist_ok=True)

    reports_dir = os.path.join(project_root, 'reports', 'lrt_reports', f'{target_digits}digits')
    os.makedirs(reports_dir, exist_ok=True)

    # Initialize the Master Blueprint File
    blueprint_path = os.path.join(lrt_data_dir, f"Sweep_Blueprint_{target_digits}digits.csv")
    with open(blueprint_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["Interval_Range", "Triplet_ID", "Prime_1", "Prime_2", "Prime_3"])

    # ================= 2. Partition 1.1x Intervals =================
    intervals = []
    current_start = 10 ** (target_digits - 1)
    upper_limit = (10 ** target_digits) - 1

    while current_start < upper_limit:
        current_end = int(current_start * 1.1)
        if current_end > upper_limit:
            current_end = upper_limit
        intervals.append((current_start, current_end))
        current_start = current_end + 1  # Add 1 to prevent overlapping bounds

    print(f"📏 Universe partitioned: Sliced into {len(intervals)} independent 1.1x micro-intervals.")

    # ================= 3. Interval-by-Interval Scanning =================
    sweep_results = []
    global_triplet_id = 1
    total_rejections = 0
    total_valid = 0

    required_primes_per_interval = triplets_per_interval * 3

    for interval_idx, (start_val, end_val) in enumerate(intervals):
        interval_id = interval_idx + 1
        print(f"\n📡 [Interval {interval_id}/{len(intervals)}] Scanning range: {start_val} - {end_val} ...")

        # Sample primes within the current geometric interval
        pool_primes = generate_ecpp_primes_for_interval(start_val, end_val, required_primes_per_interval, rng)

        # Circuit Breaker
        if len(pool_primes) < 3:
            print(f"  ⚠️ Prime Desert! Only {len(pool_primes)} primes found. Skipping interval.")
            sweep_results.append({
                "Interval_ID": interval_id, "Start": start_val, "End": end_val,
                "Valid_Triplets": 0, "Rejections": 0, "Rejection_Rate": None
            })
            continue

        # Sort and assemble into triplets
        sorted_primes = sorted(pool_primes)
        triplets_list = [tuple(sorted_primes[i: i + 3]) for i in range(0, len(sorted_primes) - 2, 3)]

        print(f"  🎯 Successfully assembled {len(triplets_list)} triplets. Dispatching to C program court...")

        interval_rejections = 0
        interval_valid = 0

        # Execute C Simulation & LRT
        for triplet in triplets_list:

            # Log this specific triplet to the Master Blueprint immediately
            with open(blueprint_path, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([f"{start_val}-{end_val}", global_triplet_id, triplet[0], triplet[1], triplet[2]])

            csv_metrics_path = os.path.join(sim_data_dir, f"triplet_{global_triplet_id:04d}_data.csv")
            report_path = os.path.join(reports_dir, f"triplet_{global_triplet_id:04d}_report.txt")

            with tempfile.TemporaryDirectory() as temp_dir:
                temp_input_path = os.path.join(temp_dir, "temp_primes.txt")
                temp_pure_path = os.path.join(temp_dir, "temp_pure.csv")

                with open(temp_input_path, 'w', encoding='utf-8') as f:
                    for prime in triplet: f.write(f"{prime}\n")

                try:
                    subprocess.run(
                        [c_exec_path, temp_input_path, temp_pure_path, csv_metrics_path, str(num_simulations)],
                        check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
                    )
                except subprocess.CalledProcessError:
                    global_triplet_id += 1
                    continue

            try:
                df = pd.read_csv(csv_metrics_path)
                df = df[df['trials'] != 'FAILED']
                df['trials'] = pd.to_numeric(df['trials'])
                agg_df = df.groupby('prime')['trials'].mean().reset_index().sort_values(by='prime').reset_index(
                    drop=True)

                if len(agg_df) == 3:
                    y_bars = agg_df['trials'].tolist()
                    primes = agg_df['prime'].tolist()
                    T_stat, p_value, p_hat, p_tilde = perform_lrt(num_simulations, y_bars)

                    is_rejected = p_value < 0.05
                    if is_rejected:
                        interval_rejections += 1
                        total_rejections += 1
                    interval_valid += 1
                    total_valid += 1

                    # Write individual Triplet Report
                    with open(report_path, 'w', encoding='utf-8') as f:
                        f.write(
                            f"Isotonic LRT Report for Triplet {global_triplet_id:04d} (Interval {start_val}-{end_val})\n")
                        f.write("=" * 50 + "\n")
                        f.write(f"Target Universe: {target_digits}-digits\n")
                        f.write(f"Sample Size (per prime): n = {num_simulations}\n\n")
                        f.write("--- Empirical Data (Ordered x1 < x2 < x3) ---\n")
                        for idx, (pr, y) in enumerate(zip(primes, y_bars)):
                            f.write(f"Prime {idx + 1}: {pr} | Average Trials (y_bar) = {y:.2f}\n")
                        f.write("\n--- Statistical Inference ---\n")
                        f.write(f"Likelihood Ratio Statistic (T) : {T_stat:.6f}\n")
                        f.write(f"Asymptotic P-Value             : {p_value:.6e}\n")
                        f.write(f"Verdict: {'REJECT H0' if is_rejected else 'FAIL TO REJECT H0'}\n")

            except Exception:
                pass

            global_triplet_id += 1

        # Record macroscopic data for the current interval
        sweep_results.append({
            "Interval_ID": interval_id,
            "Start": start_val,
            "End": end_val,
            "Valid_Triplets": interval_valid,
            "Rejections": interval_rejections,
            "Rejection_Rate": round((interval_rejections / interval_valid) * 100, 2) if interval_valid > 0 else None
        })

    # ================= 4. Export Radar Sweep Master Report =================

    # 💡 Calculate and append the Global Summary row before saving the CSV
    global_rate = round((total_rejections / total_valid) * 100, 2) if total_valid > 0 else None
    sweep_results.append({
        "Interval_ID": "GLOBAL_SUMMARY",
        "Start": "ALL",
        "End": "ALL",
        "Valid_Triplets": total_valid,
        "Rejections": total_rejections,
        "Rejection_Rate": global_rate
    })

    sweep_df = pd.DataFrame(sweep_results)
    sweep_report_path = os.path.join(reports_dir, f"Sweep_Report_{target_digits}digits.csv")
    sweep_df.to_csv(sweep_report_path, index=False)

    print("\n" + "=" * 50)
    print(f"🎉 {target_digits}-digit Universe Sweep successfully completed!")
    print(f"📊 Processed {total_valid} valid triplet tests.")
    if total_valid > 0:
        print(f"🚨 Global Rejection Rate: {global_rate}%")
    print(f"📈 Detailed 1.1x slice heatmap data saved to: {sweep_report_path}")
    print("=" * 50 + "\n")


if __name__ == "__main__":
    run_pipeline(
        target_digits=3,
        triplets_per_interval=1,
        num_simulations=10,
        seed_value=None
    )