import pandas as pd
import subprocess
import os
import time


def run_convergence_experiment(primes_file, ground_truth_file, c_executable='/Users/catgpt/Documents/ECPP_Logic_AI/01_src/pomerance'):
    print("=== Starting Convergence Benchmark ===")

    # 1. Load Ground Truth Data
    if not os.path.exists(ground_truth_file):
        print(f"[Error] Ground truth file not found: {ground_truth_file}")
        return

    truth_df = pd.read_csv(ground_truth_file)
    # Build a dictionary for fast O(1) lookups: {prime: exact_probability}
    truth_map = dict(zip(truth_df['prime'], truth_df['exact_probability']))
    print(f"Successfully loaded ground truth data for {len(truth_map)} primes.")

    # Create a directory to store all intermediate and final results
    output_dir = "/Users/catgpt/Documents/ECPP_Logic_AI/reports/p_stats/experiment_results"
    os.makedirs(output_dir, exist_ok=True)
    print(f"All intermediate data will be saved to the '{output_dir}' directory.")

    # Added 200 to test the long-tail convergence
    sample_sizes = [5, 20, 50, 100, 200]
    results_summary = []

    for n in sample_sizes:
        print(f"\n[Testing] Target sample size n = {n} ...")

        # Define permanent output files for this specific sample size
        pure_csv_path = os.path.join(output_dir, f"triples_pure_n_{n}.csv")
        metrics_csv_path = os.path.join(output_dir, f"triples_metrics_n_{n}.csv")
        detailed_stats_path = os.path.join(output_dir, f"detailed_stats_n_{n}.csv")

        start_time = time.time()

        # 3. Call the underlying C program
        try:
            subprocess.run(
                [c_executable, primes_file, pure_csv_path, metrics_csv_path, str(n)],
                check=True,
                stdout=subprocess.DEVNULL  # Hide standard output to keep the terminal clean
            )
        except subprocess.CalledProcessError as e:
            print(f"[Error] C program execution failed for n={n}: {e}")
            continue

        # 4. Read the metrics output from the C program
        if not os.path.exists(metrics_csv_path):
            print(f"[Error] C program output file not found: {metrics_csv_path}")
            continue

        metrics_df = pd.read_csv(metrics_csv_path)

        # Clean data: remove FAILED records
        valid_metrics_df = metrics_df[metrics_df['trials'] != 'FAILED'].copy()
        valid_metrics_df['trials'] = valid_metrics_df['trials'].astype(int)

        # 5. Calculate empirical probability (p_hat) and relative error for EACH prime
        detailed_records = []
        errors = []

        grouped = valid_metrics_df.groupby('prime')['trials'].sum().reset_index()

        for _, row in grouped.iterrows():
            p = int(row['prime'])
            total_trials = int(row['trials'])

            # MLE Estimate
            p_hat = n / total_trials

            if p in truth_map:
                p_exact = truth_map[p]
                relative_error = abs(p_hat - p_exact) / p_exact
                errors.append(relative_error)

                # Record the intermediate calculation steps for this prime
                detailed_records.append({
                    'prime': p,
                    'target_n': n,
                    'total_trials': total_trials,
                    'p_exact': p_exact,
                    'p_hat': p_hat,
                    'relative_error_percent': round(relative_error * 100, 4)
                })

        # Save the detailed stats for this specific 'n'
        if detailed_records:
            detailed_df = pd.DataFrame(detailed_records)
            detailed_df.to_csv(detailed_stats_path, index=False)
            print(f"  -> Detailed stats saved to: {detailed_stats_path}")

        # 6. Aggregate statistics
        if errors:
            mean_relative_error = sum(errors) / len(errors)
            elapsed_time = time.time() - start_time

            print(f"  -> Done! Time elapsed: {elapsed_time:.2f} seconds")
            print(f"  -> Mean Relative Error (MRE): {mean_relative_error * 100:.2f}%")

            results_summary.append({
                'Sample Size (n)': n,
                'Mean Relative Error (%)': round(mean_relative_error * 100, 2),
                'Time (s)': round(elapsed_time, 2)
            })

    # 7. Output the final convergence report
    print("\n=======================================")
    print("       Final Error Convergence Report")
    print("=======================================")
    report_df = pd.DataFrame(results_summary)
    print(report_df.to_string(index=False))

    # Save the master summary report
    report_file = os.path.join(output_dir, "convergence_summary_report.csv")
    report_df.to_csv(report_file, index=False)
    print(f"\n[Success] Full report and all intermediate data saved to folder: '{output_dir}'")


if __name__ == "__main__":
    # Ensure these paths match your actual files
    PRIMES_TXT = "/Users/catgpt/Documents/ECPP_Logic_AI/data/final_data/sampled_primes_input.txt"
    GROUND_TRUTH_CSV = "/Users/catgpt/Documents/ECPP_Logic_AI/data/final_data/sampled_primes_ground_truth.csv"
    C_EXECUTABLE = "/Users/catgpt/Documents/ECPP_Logic_AI/01_src/pomerance"

    run_convergence_experiment(PRIMES_TXT, GROUND_TRUTH_CSV, C_EXECUTABLE)