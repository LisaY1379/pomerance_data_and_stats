import os
import csv
import numpy as np
import pandas as pd
import time
from sklearn.isotonic import IsotonicRegression
from joblib import Parallel, delayed


# ================= Core Algorithm Module =================

def geometric_log_likelihood(n, y_bar, p):
    """Log-likelihood function for the Geometric distribution"""
    # Restrict p bounds to prevent math domain error or -inf
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return n * np.log(p) + n * (y_bar - 1) * np.log(1 - p)


def run_pava_for_geometric(y_bars, n_array):
    """
    [ULTRA-FAST VERSION] Isotonic Regression (PAVA) using scikit-learn.
    Constraint direction: increasing=True (H0 assumes trials increase as primes get larger).
    """
    # Instantiate: Explicitly set to 'monotonically increasing' constraint for trials
    ir = IsotonicRegression(increasing=True, out_of_bounds='clip')

    # Construct dummy X-axis features (order: 0, 1, 2... k-1)
    x = np.arange(len(y_bars))

    # Fit the algorithm and output the pooled y_bars directly
    y_pooled = ir.fit_transform(x, y_bars, sample_weight=n_array)

    # Convert back to Geometric distribution probability (p_tilde)
    return 1.0 / y_pooled


def calculate_lrt_statistic(n_array, y_bars):
    """
    Calculate the LRT statistic T for testing FOR Order Restriction.
    H0: Isotonic (Difficulty/trials strictly monotonically increasing or flat)
    H1: Unconstrained (Wild fluctuations, completely random patterns)
    """
    k = len(y_bars)

    # Unconstrained probability MLE (p_hat) - fits the raw data perfectly (H1)
    p_hat = 1.0 / np.array(y_bars)

    # Isotonically constrained probability MLE (p_tilde) - strictly increasing (H0)
    p_tilde = run_pava_for_geometric(y_bars, n_array)

    # Global MLE under the Least Favorable Configuration (LFC) of H0 - completely flat
    global_y_bar = np.sum(np.array(n_array) * np.array(y_bars)) / np.sum(n_array)
    p_0 = np.full(k, 1.0 / global_y_bar)

    # 1. Log-likelihood L (Unconstrained H1)
    l_unconstrained = np.sum([geometric_log_likelihood(n_array[i], y_bars[i], p_hat[i]) for i in range(k)])

    # 2. Log-likelihood L (Isotonic Constrained H0)
    l_constrained = np.sum([geometric_log_likelihood(n_array[i], y_bars[i], p_tilde[i]) for i in range(k)])

    # Core modification: T measures how much better the Unconstrained model is compared to the Isotonic model.
    # T = 2 * ( L(Unconstrained) - L(Isotonic) )
    T_stat = 2 * (l_unconstrained - l_constrained)
    if T_stat < 1e-8: T_stat = 0.0

    return T_stat, p_tilde, p_0[0]


# ================= Parallel Monte Carlo Engine =================

def single_simulation(n_array_np, p_0):
    """A single atomic Monte Carlo iteration, separated for parallelization."""
    simulated_failures = np.random.negative_binomial(n_array_np, p_0)
    simulated_total_trials = simulated_failures + n_array_np
    simulated_y_bars = simulated_total_trials / n_array_np

    T_sim, _, _ = calculate_lrt_statistic(n_array_np, simulated_y_bars)
    return T_sim


def monte_carlo_p_value(n_array, p_0, observed_T, num_simulations=10000):
    """
    Simulates the null distribution (H0) using all available CPU cores via joblib.
    """
    print(f"⏳ Firing up all CPU cores for {num_simulations} parallel Monte Carlo simulations...")
    n_array_np = np.array(n_array)

    # Parallelize the workload across all CPU threads (n_jobs=-1)
    simulated_T_stats = Parallel(n_jobs=-1, batch_size="auto")(
        delayed(single_simulation)(n_array_np, p_0) for _ in range(num_simulations)
    )

    simulated_T_stats = np.array(simulated_T_stats)

    # P-value = proportion of simulated T stats >= observed T
    p_value = np.sum(simulated_T_stats >= observed_T) / num_simulations
    return p_value, simulated_T_stats


# ================= Main Control Flow =================

def run_global_lrt(input_csv, digits, num_simulations=10000):
    if not os.path.exists(input_csv):
        print(f"❌ Error: File not found {input_csv}")
        return

    print(f"\n📂 Reading full dataset: {input_csv}")

    df = pd.read_csv(input_csv)
    df['trials'] = df['trials'].astype(str)
    df = df[df['trials'].str.upper() != 'FAILED']
    df['trials'] = pd.to_numeric(df['trials'])
    df['prime'] = pd.to_numeric(df['prime'])

    # Group, aggregate and strictly sort by prime
    agg_df = df.groupby('prime')['trials'].agg(['mean', 'count']).reset_index()
    agg_df = agg_df.sort_values(by='prime').reset_index(drop=True)

    k = len(agg_df)
    print(f"✅ Successfully loaded aggregated data for {k} primes.")
    print("🚀 Running Test FOR Order Restriction (H0: Monotonically Increasing Difficulty)")

    primes = agg_df['prime'].tolist()
    y_bars = agg_df['mean'].tolist()
    n_array = agg_df['count'].tolist()

    start_time = time.time()

    # 1. Real Data T Statistic
    observed_T, p_tilde, p_0 = calculate_lrt_statistic(n_array, y_bars)

    # 2. Parallel Monte Carlo P-Value
    p_value, _ = monte_carlo_p_value(n_array, p_0, observed_T, num_simulations)

    end_time = time.time()

    # ================= Output Report =================
    report_file = f"Global_LRT_Report_{digits}_digits.txt"
    is_rejected = p_value < 0.05

    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("=== Isotonic Likelihood Ratio Test (Test FOR Order Restriction) ===\n")
        f.write(f"Data Source: {input_csv}\n")
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

        # Inverted conclusion logic specific to "Test For Order Restriction"
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

    print("\n" + "=" * 65)
    print(f"🎉 Global LRT Analysis Complete! Time elapsed: {end_time - start_time:.2f} seconds.")
    print(f"  - Observed T Statistic : {observed_T:.4f}")
    print(f"  - Monte Carlo P-Value  : {p_value:.6e}")
    if is_rejected:
        print(f"  - Conclusion           : REJECT H0 (The trend is NOT purely monotonically increasing)")
    else:
        print(f"  - Conclusion           : FAIL TO REJECT H0 (Consistent with increasing difficulty)")
    print(f"📄 Detailed report saved to: {report_file}")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    for digits in range(10, 14):
        target_csv = f"../data/final_data/prime_{digits}_digits.csv"
        run_global_lrt(target_csv, digits, num_simulations=10000)