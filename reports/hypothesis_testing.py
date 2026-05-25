import os
import pandas as pd
import numpy as np
from scipy.stats import chi2


def run_lrt_batch_analysis(csv_path, output_dir='lrt_reports'):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if not os.path.exists(csv_path):
        print(f"Error: The file '{csv_path}' does not exist.")
        return

    df = pd.read_csv(csv_path)

    df['prime'] = pd.to_numeric(df['prime'], errors='coerce')
    df['trials'] = pd.to_numeric(df['trials'], errors='coerce')
    df = df.dropna(subset=['prime', 'trials'])

    df['prime'] = df['prime'].astype(np.int64)
    df['trials'] = df['trials'].astype(int)

    df['prime_str'] = df['prime'].astype(str)
    df['digit_length'] = df['prime_str'].str.len()

    grouped_by_length = df.groupby('digit_length')

    print(f"========== Found {len(grouped_by_length)} distinct digit segments in the data ==========\n")

    for length, group_df in grouped_by_length:
        print(f"[*] Analyzing {length} digit segment... (Target: {length}-digit primes)...")

        grouped = group_df.groupby('prime')['trials'].apply(list).to_dict()

        valid_primes = []
        trials_list = []

        for prime, trials in grouped.items():
            if len(trials) == 5:
                valid_primes.append(prime)
                trials_list.append(trials)

        if len(trials_list) < 2:
            print(f"    -> [Skipped] This numerical segment contains fewer than 2 valid prime numbers; statistical testing cannot be performed.\n")
            continue

        trials_matrix = np.array(trials_list)
        N, k = trials_matrix.shape

        S_i = np.sum(trials_matrix, axis=1)
        S_total = np.sum(S_i)

        theta_i_hat = k / (S_i + k)
        l1 = np.sum(np.where(S_i > 0, S_i * np.log(1 - theta_i_hat), 0) + k * np.log(theta_i_hat))

        theta_0_hat = (N * k) / (S_total + N * k)
        l0 = S_total * np.log(1 - theta_0_hat) + (N * k) * np.log(theta_0_hat)

        D = 2 * (l1 - l0)
        df_degrees = N - 1
        p_value = chi2.sf(D, df_degrees)

        min_trials_prime = valid_primes[np.argmin(S_i)]
        max_trials_prime = valid_primes[np.argmax(S_i)]

        report_path = os.path.join(output_dir, f'lrt_report_{length}_digits.txt')

        report_content = f"""================================================================================
STATISTICAL INFERENCE REPORT: HOMOGENEITY OF {length}-DIGIT POMERANCE TRIPLES
================================================================================

1. EXECUTIVE SUMMARY
--------------------------------------------------------------------------------
This report evaluates the distribution behavior of the numbers of computational 
trials required to find Pomerance Triples specifically within the {length}-digit 
prime interval. Assuming that the individual trials follow an i.i.d. Geometric 
distribution conditioned on a fixed success probability (theta), a Likelihood 
Ratio Test (LRT) was conducted to assess whether all primes of this magnitude 
share a uniform difficulty profile.

2. EXPERIMENT DESIGN & DATA MANIFEST
--------------------------------------------------------------------------------
- Target Scale             : {length}-Digit Primes
- Total Distinct Primes (N): {N}
- Observations per Prime (k): {k} (Independent runs)
- Total Trial Sample Size  : {N * k}
- Empirical Global Trials  : {S_total} iterations

3. HYPOTHESIS FORMULATION
--------------------------------------------------------------------------------
- Null Hypothesis (H0)     : theta_1 = theta_2 = ... = theta_N = theta_0
                             (The success probability is homogeneous across all 
                             {length}-digit primes in this sample.)

- Alternative Hypothesis(H1): At least one theta_i differs.
                             (The success probability is heterogeneous, driven 
                             by individual elliptic curve/group structures.)

4. PARAMETER ESTIMATION & TEST STATISTIC
--------------------------------------------------------------------------------
- Est. Global Success Rate (H0 theta)  : {theta_0_hat:.8f}
- Log-Likelihood under H0 (l0)         : {l0:.4f}
- Log-Likelihood under H1 (l1)         : {l1:.4f}
- Likelihood Ratio Statistic (D)       : {D:.4f}
- Degrees of Freedom (df)              : {df_degrees}
- Asymptotic P-value                   : {p_value:.4e}

5. STATISTICAL INFERENCE & INTERPRETATION
--------------------------------------------------------------------------------
"""
        if p_value < 0.05:
            report_content += f"""CONCLUSION: [ REJECT H0 ]
At a standard significance level (alpha = 0.05), the null hypothesis is STRONGLY 
REJECTED. The mathematical evidence confirms that success probabilities (theta) 
are NOT uniform across this {length}-digit block. 

Interpretation:
Even though these primes are of the same numerical magnitude, the computational 
variance cannot be ascribed to random geometric noise alone. Distinct prime fields 
significantly alter the density of smooth 2-Sylow orders in their respective Hasse 
intervals, making certain {length}-digit structures inherently more efficient to 
solve than others.

Appendix Notes:
- Most efficient prime in sample: p = {min_trials_prime} (Total trials = {np.min(S_i)})
- Least efficient prime in sample: p = {max_trials_prime} (Total trials = {np.max(S_i)})
"""
        else:
            report_content += f"""CONCLUSION: [ FAIL TO REJECT H0 ]
The test FAILS TO REJECT the null hypothesis at alpha = 0.05. There is no 
statistically significant evidence indicating that success rates differ between 
individual {length}-digit primes in this sample.

Interpretation:
The variance in trials observed between these primes is statistically consistent 
with pure geometric fluctuations around a single, shared success rate parameter 
(theta_0 = {theta_0_hat:.6f}). The numerical thickness of the Hasse boundaries 
remains homogeneously balanced across this specific interval space.

Appendix Notes:
- Most efficient prime in sample: p = {min_trials_prime} (Total trials = {np.min(S_i)})
- Least efficient prime in sample: p = {max_trials_prime} (Total trials = {np.max(S_i)})
"""

        report_content += """--------------------------------------------------------------------------------
REPORT END — Prepared for Infrastructure Characterization and Metric Distribution
================================================================================"""

        # 写入文件
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_content)

        print(f"    -> [Success] Generated report'{report_path}'")

    print("\n========== Analysis done ==========")

# --- EXECUTION ENGINE ---
if __name__ == "__main__":
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    PARENT_DIR = os.path.dirname(current_script_dir)
    csv_file_target = os.path.join(PARENT_DIR, 'final_data', 'triples_metrics_3to6digits_seed1778732485.csv')
    output_directory = os.path.join(current_script_dir, 'lrt_reports')
    run_lrt_batch_analysis(csv_file_target, output_dir=output_directory)