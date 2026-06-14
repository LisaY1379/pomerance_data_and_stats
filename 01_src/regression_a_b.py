import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import os


def run_regression_analysis(csv_file, output_report_file="regression_summary_report.txt"):
    print("=== Loading and Preprocessing Data ===")

    if not os.path.exists(csv_file):
        print(f"[Error] Data file not found: {csv_file}")
        return

    df = pd.read_csv(csv_file)

    # 1. Group by prime and calculate the mean trials (Estimating 1 / success_rate)
    df_agg = df.groupby('prime').agg({
        'trials': 'mean',
        'param_a_count': 'first',
        'class_number': 'first'
    }).reset_index()

    # 2. Calculate the independent variable: sqrt(p)
    df_agg['sqrt_p'] = np.sqrt(df_agg['prime'])

    # Replace 0 with a very small number to prevent log(0) errors
    df_agg['class_number'] = df_agg['class_number'].replace(0, 1e-5)

    # Take logarithms for the Log-Log model
    df_agg['log_trials'] = np.log(df_agg['trials'])
    df_agg['log_sqrt_p'] = np.log(df_agg['sqrt_p'])
    df_agg['log_a'] = np.log(df_agg['param_a_count'])
    df_agg['log_b'] = np.log(df_agg['class_number'])

    num_samples = len(df_agg)
    print(f"Aggregated data resulting in {num_samples} unique prime samples.\n")

    # Open a text file to write our report
    with open(output_report_file, 'w', encoding='utf-8') as f:
        f.write("==============================================================================\n")
        f.write("                 ECPP Complexity Regression Analysis Report                   \n")
        f.write("==============================================================================\n")
        f.write(f"Source Data File: {csv_file}\n")
        f.write(f"Total Unique Primes Analyzed: {num_samples}\n\n")

        # =================================================================
        # Model 1: Standard Multiple Linear Regression (OLS)
        # =================================================================
        print("=== Model 1: Standard Multiple Linear Regression (OLS) ===")
        model_linear = smf.ols('trials ~ sqrt_p + param_a_count + class_number', data=df_agg).fit()
        print(model_linear.summary())

        f.write("=== Model 1: Standard Multiple Linear Regression (OLS) ===\n")
        f.write("Equation: trials = beta0 + beta1 * sqrt_p + beta2 * a + beta3 * b\n")
        f.write(model_linear.summary().as_text())
        f.write("\n\n" + "=" * 80 + "\n\n")

        # =================================================================
        # Model 2: Semi-Log Linear Regression (Log-Linear)
        # =================================================================
        print("\n=== Model 2: Semi-Log Linear Regression (Log-Linear) ===")
        model_log_linear = smf.ols('log_trials ~ sqrt_p + param_a_count + class_number', data=df_agg).fit()
        print(model_log_linear.summary())

        f.write("=== Model 2: Semi-Log Linear Regression (Log-Linear) ===\n")
        f.write("Equation: log(trials) = beta0 + beta1 * sqrt_p + beta2 * a + beta3 * b\n")
        f.write(model_log_linear.summary().as_text())
        f.write("\n\n" + "=" * 80 + "\n\n")

        # =================================================================
        # Model 3: Log-Log Linear Regression (Highly Recommended)
        # =================================================================
        print("\n=== Model 3: Log-Log Linear Regression ===")
        model_log_log = smf.ols('log_trials ~ log_sqrt_p + log_a + log_b', data=df_agg).fit()
        print(model_log_log.summary())

        # Print exact p-values to the console
        print("\n=== Exact P-Values for Model 3 ===")
        print(model_log_log.pvalues)

        f.write("=== Model 3: Log-Log Linear Regression ===\n")
        f.write("Equation: log(trials) = beta0 + beta1 * log(sqrt_p) + beta2 * log(a) + beta3 * log(b)\n")
        f.write(model_log_log.summary().as_text())

        # Add the exact p-values to the text report
        f.write("\n\n--- Exact Significance Test (P-Values) for Model 3 ---\n")
        f.write("Note: Values extremely close to 0 confirm highly significant impact.\n\n")

        # Format the p-values in scientific notation (e.g., 1.23e-45)
        for param, pval in model_log_log.pvalues.items():
            f.write(f"{param:<15}: {pval:.4e}\n")

        f.write("\n==============================================================================\n")

    print(f"\n[Success] Full regression report successfully saved to: {output_report_file}")


if __name__ == "__main__":
    # Ensure this points to your fully concatenated CSV file
    COMBINED_DATA_CSV = "/Users/catgpt/Documents/ECPP_Logic_AI/data/final_data/all_triples_combined.csv"

    # You can customize the output report name and path here
    OUTPUT_REPORT_TXT = "/Users/catgpt/Documents/ECPP_Logic_AI/data/final_data/regression_summary_report.txt"

    run_regression_analysis(COMBINED_DATA_CSV, OUTPUT_REPORT_TXT)