import os
import sys
import glob
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from datetime import datetime

# 🌟 Set matplotlib to headless 'Agg' backend for silent UI-free rendering in PyCharm
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt


# =========================================================================
# ⚙️ MULTI-SLICE AUTOMATIC DATA INGESTOR
# =========================================================================

def ingest_regression_data(targets=None, date_str=None):
    """
    Dynamically tracks, loads, and concatenates data slices based on the target set.
    Supports flexible elements: plain integers for baseline, or tuples for versioned runs.
    Returns: (Combined DataFrame, list of descriptive path tokens, sandbox_type)
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_data_dir = os.path.abspath(os.path.join(script_dir, "..", "..", "data"))
    historical_dir = os.path.join(base_data_dir, "historical_runs")

    # Scenario A: Default Baseline Database (No targets passed)
    if targets is None:
        combined_file = os.path.join(base_data_dir, "all_triples_combined_precomputed.csv")
        if not os.path.exists(combined_file):
            print(f"❌ Error: Combined baseline database not found at {combined_file}")
            sys.exit(1)
        print(f"📂 Loading default baseline database: {combined_file}")
        return pd.read_csv(combined_file), ["baseline"], "reproduced"

    # Scenario B: Aggregated Dataset via Flexible Set Processing
    if not isinstance(targets, set) or len(targets) == 0:
        print("❌ API Usage Error: 'targets' must be a non-empty set.")
        sys.exit(1)

    if date_str is None:
        date_str = datetime.now().strftime("%Y%m%d")

    df_list = []
    tokens = []
    sandbox_type = "reproduced"

    # 🛠️ Step 1: Normalize all elements in the set into uniform (digit, version) states
    normalized_targets = []
    for item in targets:
        if isinstance(item, int):
            # User passed a plain integer like 3 -> interpret as (3, None)
            normalized_targets.append((item, None))
        elif isinstance(item, tuple):
            if len(item) == 1:
                # User passed a single-element tuple like (3,) -> interpret as (3, None)
                normalized_targets.append((item[0], None))
            elif len(item) == 2:
                # User passed a complete tuple like (3, 2)
                normalized_targets.append(item)
            else:
                print(f"❌ API Usage Error: Unsupported tuple format {item}. Use (digit, version) or (digit,).")
                sys.exit(1)
        else:
            print(f"❌ TypeError: Unsupported item type {type(item)} inside targets set.")
            sys.exit(1)

    # Sort normalized targets to maintain consistent, predictable naming tokens
    for digit, version in sorted(normalized_targets, key=lambda x: (x[0], x[1] if x[1] is not None else -1)):
        if version is None:
            target_path = os.path.join(base_data_dir, f"{digit}_digits_precomputed.csv")
            tokens.append(f"{digit}d")
        else:
            # Route to custom historical run slice and activate original_data sandbox
            sandbox_type = "original_data"
            exact_pattern = os.path.join(historical_dir, f"triples_{digit}d_s*_{date_str}_v{version}.csv")
            matches = glob.glob(exact_pattern)

            if not matches:
                wildcard_pattern = os.path.join(historical_dir, f"triples_{digit}d_s*_v{version}.csv")
                matches = glob.glob(wildcard_pattern)

            if matches:
                target_path = max(matches, key=os.path.getmtime)
            else:
                print(f"❌ Error: Slice for digit={digit}, version={version} could not be located.")
                sys.exit(1)
            tokens.append(f"{digit}dv{version}")

        print(f"📂 Accumulating matrix slice: {target_path}")
        df_list.append(pd.read_csv(target_path))

    combined_df = pd.concat(df_list, ignore_index=True)
    return combined_df, tokens, sandbox_type


# =========================================================================
# 📊 VISUALIZATION PLOTTING ENGINE (Colocated with Reports)
# =========================================================================

def generate_and_save_regression_plot(df_agg, model, output_filepath):
    """
    Generates an Actual vs. Predicted Log-Trials scatter plot stratified by categorical parameter A.
    Saves silently to the exact same directory as the text report.
    """
    plt.figure(figsize=(10, 8), dpi=300)

    # Extract fitted values and actual observed log trials
    predicted = model.fittedvalues
    actual = df_agg['log_trials']
    a_counts = df_agg['param_a_count']

    # Define a distinct academic color palette and marker mapping for categorical A values
    palette = {1: '#EF4444', 2: '#F59E0B', 3: '#10B981', 4: '#3B82F6'}
    markers = {1: 'o', 2: 's', 3: '^', 4: 'D'}

    # Plot each categorical group separately to build a clean legend
    for a_val in sorted(df_agg['param_a_count'].unique()):
        mask = (a_counts == a_val)
        plt.scatter(
            predicted[mask],
            actual[mask],
            alpha=0.6,
            s=40,
            color=palette.get(a_val, '#64748B'),
            marker=markers.get(a_val, 'o'),
            label=f'Hasse Interval Multiples ($A = {int(a_val)}$)'
        )

    # Plot the perfect fit diagonal line (y = x)
    min_val = min(predicted.min(), actual.min())
    max_val = max(predicted.max(), actual.max())
    plt.plot([min_val, max_val], [min_val, max_val], color='#475569', linestyle='--', linewidth=2,
             label=r'Perfect Fit Diagonal ($Y = \hat{Y}$)')

    # Add R^2 text box annotation
    r_squared = model.rsquared
    plt.text(
        0.05, 0.90,
        f'OLS Variance Explained:\n$R^2 = {r_squared:.4f}$ ($({r_squared * 100:.1f}\\%)$)\n$N = {len(df_agg)}$ Primes',
        transform=plt.gca().transAxes,
        fontsize=11,
        verticalalignment='top',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='#CBD5E1', alpha=0.9)
    )

    # Aesthetics and labels (Using raw strings r'...' to prevent LaTeX syntax warnings)
    plt.title("ECPP Search Difficulty Log-Log Regression Model\nActual vs. Predicted Expected Trials", fontsize=14,
              fontweight='bold', pad=15)
    plt.xlabel(r"Predicted Search Difficulty ($\log(\widehat{\text{trials}})$)", fontsize=12)
    plt.ylabel(r"Observed Empirical Difficulty ($\log(\text{trials})$)", fontsize=12)

    plt.grid(True, linestyle='--', alpha=0.3)
    plt.legend(loc='lower right', frameon=True, facecolor='white', framealpha=0.95, fontsize=10)
    plt.tight_layout()

    # Save to identical path with _plot.png extension
    plt.savefig(output_filepath)
    plt.close()


# =========================================================================
# 📈 CORE REGRESSION & DYNAMIC SANDBOXING CORE
# =========================================================================

def run_regression(targets=None, date_str=None):
    """
    Universal programmatic interface for Log-Log Complexity OLS Regression with automated plotting.
    Examples:
      run_regression()                                # Standard Baseline
      run_regression({(3, None), (3, 2), (4, 3)})    # Multi-slice aggregate
    """
    # 1. Gather combined dataset and path tokens via routing core
    df, tokens, sandbox_type = ingest_regression_data(targets, date_str)

    print("=== Loading and Preprocessing Data ===")
    df['trials'] = df['trials'].astype(str)
    df = df[df['trials'].str.upper() != 'FAILED']
    df['trials'] = pd.to_numeric(df['trials'])
    df['prime'] = pd.to_numeric(df['prime'])

    # Group by prime and calculate the mean trials (Estimating 1 / success_rate)
    df_agg = df.groupby('prime').agg({
        'trials': 'mean',
        'param_a_count': 'first',
        'class_number': 'first'
    }).reset_index()

    # Calculate the independent variable: sqrt(p)
    df_agg['sqrt_p'] = np.sqrt(df_agg['prime'])

    # Replace 0 with a very small number to prevent log(0) errors
    df_agg['class_number'] = df_agg['class_number'].replace(0, 1e-5)

    # Take logarithms for continuous variables ONLY
    df_agg['log_trials'] = np.log(df_agg['trials'])
    df_agg['log_sqrt_p'] = np.log(df_agg['sqrt_p'])
    df_agg['log_b'] = np.log(df_agg['class_number'])

    num_samples = len(df_agg)
    print(f"Aggregated data resulting in {num_samples} unique prime samples.\n")

    if num_samples < 4:
        print("❌ Critical Error: Insufficient unique degrees of freedom to fit OLS summary matrix.")
        return

    # =========================================================================
    # 📁 REPORTS DIRECTORY ROUTING & CONFLICT COLLISION RESOLUTION
    # =========================================================================
    script_dir = os.path.dirname(os.path.abspath(__file__))
    reports_base_dir = os.path.abspath(os.path.join(script_dir, "..", "..", "reports"))

    # Assign path segment depending on data type consumed
    folder_prefix = "reports_original_data" if sandbox_type == "original_data" else "reports_reproduced"
    target_subfolder = os.path.join(reports_base_dir, folder_prefix, "regression")

    if not os.path.exists(target_subfolder):
        os.makedirs(target_subfolder)

    # Assemble unique combined filename profile token array
    base_filename = f"regression_report_{'_'.join(tokens)}"

    # Auto-increment duplicate avoidance logic loop
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

    # ================= File Reporting Engine (100% Identical Text) =================
    with open(full_report_path, 'w', encoding='utf-8') as f:
        f.write("==============================================================================\n")
        f.write("                 ECPP Complexity Regression Analysis Report                   \n")
        f.write("==============================================================================\n")
        f.write(f"Source Data File: Aggregated Selection Array\n")
        f.write(f"Total Unique Primes Analyzed: {num_samples}\n\n")

        print("\n=== Advanced Model: Categorical Fixed Effects for 'a' ===")
        model_advanced = smf.ols('log_trials ~ log_sqrt_p + C(param_a_count) + log_b', data=df_agg).fit()
        print(model_advanced.summary())

        print("\n=== Exact P-Values ===")
        print(model_advanced.pvalues)

        f.write("=== Advanced Model: Log-Log Regression with Categorical 'a' ===\n")
        f.write("Equation: log(trials) = beta0 + beta1 * log(sqrt_p) + beta_a(1,2,3,4) + beta3 * log(b)\n")
        f.write(model_advanced.summary().as_text())

        f.write("\n\n--- Exact Significance Test (P-Values) ---\n")
        f.write("Note: Values extremely close to 0 confirm highly significant impact.\n\n")

        for param, pval in model_advanced.pvalues.items():
            f.write(f"{param:<30}: {pval:.4e}\n")

        f.write("\n==============================================================================\n")

    # 🌟 Trigger plot generation directly to the matching path
    generate_and_save_regression_plot(df_agg, model_advanced, output_filepath=full_plot_path)

    print(f"\n[Success] Full regression report successfully saved to: {full_report_path}")
    print(f"[Success] Regression visualization saved to       : {full_plot_path}")


if __name__ == "__main__":
    # Test execution
    print("🧪 Testing Automated Regression Report & Plot Colocation...")
    run_regression()