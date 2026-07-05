import sys
import os
import gzip

# Ensure the parent directory is added to sys.path to seamlessly import from 'output'
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

from output.filter_strategies import (
    filter_01_quadratic_residue_bias,
)


def evaluate_filter_on_pp16A(filter_func, filepath="../data/p_data/pp16A.txt"):
    """Calculates the False Negative Rate β on the pp16A dataset (adapted for p, A signature)."""
    total_true_solutions = 0
    false_negatives = 0

    # Automatically resolve absolute paths to prevent execution-directory mismatches
    full_path = os.path.join(project_root, filepath)
    if not os.path.exists(full_path):
        # Fallback: check raw relative path directly
        full_path = filepath
        if not os.path.exists(full_path):
            print(f"⚠️ Cannot find data file: {filepath}. Please check your path configuration!")
            return 0.0

    open_fn = gzip.open if full_path.endswith('.gz') else open

    print(f"Reading dataset and evaluating filter function [{filter_func.__name__}]...")

    with open_fn(full_path, 'rt', encoding='utf-8') as f:
        for line in f:
            # Core Fix 1: Replace commas with spaces to seamlessly parse CSV/TSV data layout
            cleaned_line = line.replace(',', ' ').strip()
            parts = cleaned_line.split()
            if not parts or len(parts) < 2:
                continue

            p = int(parts[0])

            # Iterate through all valid A values listed on this row for prime p
            for str_A in parts[1:]:
                A = int(str_A)
                total_true_solutions += 1

                # Core Fix 2: Updated to pass (p, A) per your new signature layout
                if not filter_func(p, A):
                    false_negatives += 1

    if total_true_solutions == 0:
        print("⚠️ Dataset opened successfully, but no valid data rows were parsed.")
        return 0.0

    beta = false_negatives / total_true_solutions
    return beta


# ==========================================================
# Run Regression Benchmarks
# ==========================================================
if __name__ == "__main__":
    # Configure your test layout: (Function Object, Theoretical Pruning Rate α)
    test_suite = [
        (filter_01_quadratic_residue_bias, 0.50),
    ]

    for filter_fn, alpha in test_suite:
        beta = evaluate_filter_on_pp16A(filter_fn)
        gamma = (1 - beta) / (1 - alpha)

        print("-" * 55)
        print(f"📊 Filter Evaluation Report: {filter_fn.__name__}")
        print(f" - Theoretical Pruning Rate α : {alpha:.4%}")
        print(f" - True Solution False Negative Rate β : {beta:.4%}")
        print(f" - Effective Density Gain Ratio γ : {gamma:.2f}x")

        if gamma > 1.0:
            print(" 👉 Conclusion: ✅ EXCELLENT ACCELERATION! True solution density is highly concentrated.")
        else:
            print(" 👉 Conclusion: ❌ INEFFECTIVE! The filter discards valid solutions proportionally with noise.")