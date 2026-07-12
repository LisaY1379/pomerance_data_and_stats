import os
import sys
import glob
import time
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.isotonic import IsotonicRegression
from joblib import Parallel, delayed

# 🌟 Set matplotlib to headless 'Agg' backend so plots save silently without UI popups in PyCharm
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt


# =========================================================================
# ⚙️ AUTOMATIC PATH RESOLVER
# =========================================================================

def resolve_file_path(digit=None, version=None, date_str=None, override_path=None):
    """
    Intelligently routes data consumption based on (digit, version) inputs.
    """
    if override_path and os.path.exists(override_path):
        return override_path

    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_data_dir = os.path.abspath(os.path.join(script_dir, "..", "..", "data"))
    historical_dir = os.path.join(base_data_dir, "historical_runs")

    if version is None:
        if digit is not None:
            digit_file = os.path.join(base_data_dir, f"{digit}_digits_precomputed.csv")
            if os.path.exists(digit_file):
                return digit_file
        combined_file = os.path.join(base_data_dir, "all_triples_combined_precomputed.csv")
        if os.path.exists(combined_file):
            return combined_file
        print(f"❌ Error: Could not locate baseline data for digit={digit} in {base_data_dir}")
        sys.exit(1)

    if digit is None:
        print("❌ API Usage Error: You must specify 'digit' when querying a specific 'version' (e.g., lrt(3, 2)).")
        sys.exit(1)

    if date_str is None:
        date_str = datetime.now().strftime("%Y%m%d")

    exact_pattern = os.path.join(historical_dir, f"triples_{digit}d_s*_{date_str}_v{version}.csv")
    matches = glob.glob(exact_pattern)

    if not matches:
        wildcard_pattern = os.path.join(historical_dir, f"triples_{digit}d_s*_v{version}.csv")
        matches = glob.glob(wildcard_pattern)

    if matches:
        return max(matches, key=os.path.getmtime)
    else:
        print(f"❌ Error: Historical slice for digit={digit}, version={version} not found in {historical_dir}")
        sys.exit(1)


# ================= Core Algorithm Module =================

def geometric_log_likelihood(n, y_bar, p):
    """Log-likelihood function for the Geometric distribution"""
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return n * np.log(p) + n * (y_bar - 1) * np.log(1 - p)


def run_pava_for_geometric(y_bars, n_array):
    """Isotonic Regression (PAVA) using scikit-learn."""
    ir = IsotonicRegression(increasing=True, out_of_bounds='clip')
    x = np.arange(len(y_bars))
    y_pooled = ir.fit_transform(x, y_bars, sample_weight=n_array)
    return 1.0 / y_pooled


def calculate_lrt_statistic(n_array, y_bars):
    """Calculate the LRT statistic T for testing FOR Order Restriction."""
    k = len(y_bars)
    p_hat = 1.0 / np.array(y_bars)
    p_tilde = run_pava_for_geometric(y_bars, n_array)
    global_y_bar = np.sum(np.array(n_array) * np.array(y_bars)) / np.sum(n_array)
    p_0 = np.full(k, 1.0 / global_y_bar)

    l_unconstrained = np.sum([geometric_log_likelihood(n_array[i], y_bars[i], p_hat[i]) for i in range(k)])
    l_constrained = np.sum([geometric_log_likelihood(n_array[i], y_bars[i], p_tilde[i]) for i in range(k)])

    T_stat = 2 * (l_unconstrained - l_constrained)
    if T_stat < 1e-8: T_stat = 0.0
    return T_stat, p_tilde, p_0[0]


# ================= Parallel Monte Carlo Engine =================

def single_simulation(n_array_np, p_0):
    """A single atomic Monte Carlo iteration."""
    simulated_failures = np.random.negative_binomial(n_array_np, p_0)
    simulated_total_trials = simulated_failures + n_array_np
    simulated_y_bars = simulated_total_trials / n_array_np
    T_sim, _, _ = calculate_lrt_statistic(n_array_np, simulated_y_bars)
    return T_sim


def monte_carlo_p_value(n_array, p_0, observed_T, num_simulations=10000):
    """Simulates the null distribution (H0) via joblib."""
    print(f"⏳ Firing up all CPU cores for {num_simulations} parallel Monte Carlo simulations...")
    n_array_np = np.array(n_array)
    simulated_T_stats = Parallel(n_jobs=-1, batch_size="auto")(
        delayed(single_simulation)(n_array_np, p_0) for _ in range(num_simulations)
    )
    simulated_T_stats = np.array(simulated_T_stats)
    p_value = np.sum(simulated_T_stats >= observed_T) / num_simulations
    return p_value, simulated_T_stats


# =========================================================================
# 📊 VISUALIZATION PLOTTING ENGINE (Colocated with Reports)
# =========================================================================

def generate_and_save_lrt_plot(primes, y_bars, p_tilde, p_0, observed_T, p_value, output_filepath):
    """
    Generates the Three-Tier Monotonic Step Overlay plot and saves it to the exact same directory as the text report.
    """
    y_tilde = 1.0 / np.array(p_tilde)
    y_baseline = 1.0 / p_0

    plt.figure(figsize=(12, 6), dpi=300)

    # 1. Unconstrained observed means (H1)
    plt.scatter(range(len(primes)), y_bars, color='#64748B', alpha=0.6, s=30,
                label=r'Observed Mean Trials ($\overline{y}$, Unconstrained $H_1$)')
    plt.plot(range(len(primes)), y_bars, color='#94A3B8', linestyle='--', alpha=0.4)

    # 2. Isotonic PAVA Step Regression (H0)
    plt.plot(range(len(primes)), y_tilde, color='#0284C7', linewidth=2.5, drawstyle='steps-post',
             label=r'Isotonic Monotonically Increasing Fit ($\widetilde{y}$, Constrained $H_0$)')

    # 3. Global Least Favorable Baseline
    plt.axhline(y=y_baseline, color='#DC2626', linestyle=':', linewidth=1.8,
                label=f'Global LFC Baseline ($y_0 = {y_baseline:.1f}$)')

    # Shading the pooling penalty area
    plt.fill_between(range(len(primes)), y_bars, y_tilde, color='#0284C7', alpha=0.15, step='post',
                     label=r'PAVA Pooling Penalty Area (Drives $T$-Stat)')

    # Aesthetics
    plt.title(
        f"Isotonic Likelihood Ratio Test (LRT) Monotonous Fit\nObserved $T = {observed_T:.2f}$ | Monte Carlo $P$-Value = {p_value:.4e}",
        fontsize=13, fontweight='bold', pad=15)
    plt.xlabel("Sorted Prime Index (Increasing Magnitude $\to$)", fontsize=11)
    plt.ylabel("Expected Search Trials", fontsize=11)

    step_ticks = max(1, len(primes) // 10)
    plt.xticks(ticks=range(0, len(primes), step_ticks),
               labels=[str(primes[i]) for i in range(0, len(primes), step_ticks)], rotation=30)

    plt.grid(True, linestyle='--', alpha=0.3)
    plt.legend(loc='upper left', frameon=True, facecolor='white', framealpha=0.9)
    plt.tight_layout()

    # Save to the requested path (identically named with _plot.png)
    plt.savefig(output_filepath)
    plt.close()


# ================= Main Control Flow =================

def lrt(digit=None, version=None, date_str=None, num_simulations=10000, input_csv=None):
    """
    Universal entry point for Isotonic LRT with dynamic sandbox directory routing and automated plotting.
    """
    target_csv = resolve_file_path(digit, version, date_str, override_path=input_csv)
    print(f"\n📂 Reading target dataset: {target_csv}")

    df = pd.read_csv(target_csv)
    df['trials'] = df['trials'].astype(str)
    df = df[df['trials'].str.upper() != 'FAILED']
    df['trials'] = pd.to_numeric(df['trials'])
    df['prime'] = pd.to_numeric(df['prime'])

    if digit is not None and 'digits' in df.columns:
        df = df[df['digits'] == digit]
    elif digit is not None and "combined" in str(target_csv):
        df = df[df['prime'].astype(str).str.len() == digit]

    agg_df = df.groupby('prime')['trials'].agg(['mean', 'count']).reset_index()
    agg_df = agg_df.sort_values(by='prime').reset_index(drop=True)

    k = len(agg_df)
    if k == 0:
        print(f"⚠️ Warning: No valid prime data found for digit={digit}. Skipping.")
        return

    print(f"✅ Successfully loaded aggregated data for {k} primes.")
    print("🚀 Running Test FOR Order Restriction (H0: Monotonically Increasing Difficulty)")

    primes = agg_df['prime'].tolist()
    y_bars = agg_df['mean'].tolist()
    n_array = agg_df['count'].tolist()

    start_time = time.time()
    observed_T, p_tilde, p_0 = calculate_lrt_statistic(n_array, y_bars)
    p_value, _ = monte_carlo_p_value(n_array, p_0, observed_T, num_simulations)
    end_time = time.time()

    # =========================================================================
    # 📁 DYNAMIC SANDBOX PATH ROUTING & AUTO-INCREMENT VERSIONING
    # =========================================================================
    script_dir = os.path.dirname(os.path.abspath(__file__))
    reports_base_dir = os.path.abspath(os.path.join(script_dir, "..", "..", "reports"))

    if version is None:
        target_subfolder = os.path.join(reports_base_dir, "reports_reproduced", "lrt")
        digit_label = digit if digit is not None else "All"
        base_filename = f"Global_LRT_Report_{digit_label}_digits"
    else:
        target_subfolder = os.path.join(reports_base_dir, "reports_original_data", "lrt")
        base_filename = f"Global_LRT_Reports_v{version}"

    if not os.path.exists(target_subfolder):
        os.makedirs(target_subfolder)

    file_counter = 1
    while True:
        if file_counter == 1:
            candidate_filename = f"{base_filename}.txt"
        else:
            candidate_filename = f"{base_filename}_{file_counter}.txt"

        full_report_path = os.path.join(target_subfolder, candidate_filename)

        if not os.path.exists(full_report_path):
            break
        file_counter += 1

    # 🌟 Derive the matching plot filepath by stripping .txt and appending _plot.png
    root_path, _ = os.path.splitext(full_report_path)
    full_plot_path = f"{root_path}_plot.png"

    # ================= Output Report Generation =================
    is_rejected = p_value < 0.05

    with open(full_report_path, 'w', encoding='utf-8') as f:
        f.write("=== Isotonic Likelihood Ratio Test (Test FOR Order Restriction) ===\n")
        f.write(f"Data Source: {target_csv}\n")
        f.write(f"Test Dimension (k): {k} primes\n")
        f.write(f"Monte Carlo Simulations: {num_simulations}\n")
        f.write(f"Global Base Success Rate (LFC p_0): {p_0:.6f}\n")
        f.write("-" * 65 + "\n")
        f.write("Hypothesis Frame:\n")
        f.write("  H0: Search difficulty (trials) is monotonically increasing or flat.\n")
        f.write("  H1: Search difficulty is completely unconstrained (wild fluctuations).\n")
        f.write("-" * 65 + "\n")
        f.write(f"Observed LRT Statistic (T) : {observed_T:.6f}\n")
        f.write(f"Monte Carlo P-Value        : {p_value:.6e}\n")

        if is_rejected:
            f.write(
                "Final Statistical Verdict  : REJECT H0 (Data fluctuates too wildly; strictly increasing difficulty is rejected)\n")
        else:
            f.write(
                "Final Statistical Verdict  : FAIL TO REJECT H0 (Data is statistically consistent with a monotonically increasing trend)\n")

        f.write("-" * 65 + "\n")
        f.write(f"Computation Time: {end_time - start_time:.2f} seconds\n\n")

        f.write("--- Isotonically Smoothed Inference Parameters (p_tilde) for Each Prime ---\n")
        for i in range(k):
            f.write(
                f"Prime: {primes[i]:<15} | Original Mean y_bar: {y_bars[i]:>8.2f} | Fitted Prob p_tilde: {p_tilde[i]:.6f} | Sample Size n: {n_array[i]}\n")

    # 🌟 Trigger plot generation directly to the matching path
    generate_and_save_lrt_plot(primes, y_bars, p_tilde, p_0, observed_T, p_value, output_filepath=full_plot_path)

    print("\n" + "=" * 65)
    print(f"🎉 Global LRT Analysis Complete! Time elapsed: {end_time - start_time:.2f} seconds.")
    print(f"  - Observed T Statistic : {observed_T:.4f}")
    print(f"  - Monte Carlo P-Value  : {p_value:.6e}")
    if is_rejected:
        print(f"  - Conclusion           : REJECT H0 (The trend is NOT purely monotonically increasing)")
    else:
        print(f"  - Conclusion           : FAIL TO REJECT H0 (Consistent with increasing difficulty)")
    print(f"📄 Detailed report saved to: {full_report_path}")
    print("📈 Visualization saved to  : {full_plot_path}")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    # Test execution
    print("🧪 Testing Automated Report & Plot Colocation...")
    lrt(3)