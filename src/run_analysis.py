import os
import sys
import time

# =========================================================================
# 📦 MODULE IMPORTS (Ingesting core execution engines from submodules)
# =========================================================================
try:
    # Assuming 'lrt_pipeline.py' provides the 'lrt()' function
    from src.core.lrt_pipeline import lrt as _execute_lrt
    # Assuming 'regression_a_b.py' provides the 'run_regression()' function
    from src.core.regression_a_b import run_regression as _execute_regression
except ImportError as e:
    print(f"❌ Critical Error: Failed to import analytical submodules. Details: {e}")
    print("   Please ensure 'lrt_pipeline.py' and 'regression_a_b.py' reside in the same 'src/' directory.")
    sys.exit(1)


# =========================================================================
# 🛠️ EXPOSED PUBLIC API (Exposing clean interfaces for global invocation)
# =========================================================================

def lrt(digit=None, version=None, date_str=None, num_simulations=10000, input_csv=None):
    """
    Exposed wrapper for Isotonic LRT. Directly interfaces with lrt_pipeline.lrt().
    Example usage in PyCharm console or custom scripts:
        from run_analysis import lrt
        lrt(9)
        lrt(9, 2)
    """
    return _execute_lrt(
        digit=digit,
        version=version,
        date_str=date_str,
        num_simulations=num_simulations,
        input_csv=input_csv
    )


def regression(targets=None, date_str=None):
    """
    Exposed wrapper for Log-Log OLS Regression. Directly interfaces with regression_a_b.run_regression().
    Example usage in PyCharm console or custom scripts:
        from run_analysis import regression
        regression()
        regression({10, 11, 12, 13})
    """
    return _execute_regression(targets=targets, date_str=date_str)


# =========================================================================
# 🚀 MASTER REPRODUCTION ENGINE (One-Click Pipeline Orchestrator)
# =========================================================================

def reproduce():
    """
    The ultimate end-to-end analytical orchestrator.
    Executes the exact replication suite specified in the README without overwriting baseline assets.
    """
    total_start_time = time.time()
    print("==============================================================================")
    print("🚀 MASTER STATISTICAL REPRODUCTION SEQUENCE ACTIVATED")
    print("==============================================================================")
    print("This pipeline will automatically generate all baselines, LRT models, and OLS matrices.")
    print("All output artifacts will be safely isolated inside '../reports/reports_reproduced/'.")
    print("------------------------------------------------------------------------------\n")

    # -------------------------------------------------------------------------
    # Stage 1: Isotonic LRT across ALL individual prime scales (3 to 13 Digits)
    # -------------------------------------------------------------------------
    print(">>> [STAGE 1 / 4] Executing Isotonic LRT for 3 to 13 Digit Cohorts...")
    for d in range(3, 14):
        print(f"\n--- Analyzing Monotonicity for Scale: {d} Digits ---")
        # Target the precomputed specific filename mapping convention requested
        precomputed_file = f"../data/{d}_digits_precomputed.csv"

        if os.path.exists(precomputed_file):
            lrt(digit=d, input_csv=precomputed_file)
        else:
            # Fall back gracefully to the standard internal file routing database if specific precomputed is absent
            print(f"  [i] Note: '{precomputed_file}' not found. Relying on default database routing...")
            lrt(digit=d)

    # -------------------------------------------------------------------------
    # Stage 2: Global Baseline Regression (All Precomputed Data Combined)
    # -------------------------------------------------------------------------
    print("\n------------------------------------------------------------------------------")
    print(">>> [STAGE 2 / 4] Fitting Global Baseline Log-Log OLS Regression Model...")
    print("------------------------------------------------------------------------------")
    # Passing empty arguments invokes the full combined default database core routing
    regression()

    # -------------------------------------------------------------------------
    # Stage 3: High-Digit Compound Slice Regression (10 to 13 Digits)
    # -------------------------------------------------------------------------
    print("\n------------------------------------------------------------------------------")
    print(">>> [STAGE 3 / 4] Executing Compound Regression: 10 to 13 Digits Set...")
    print("------------------------------------------------------------------------------")
    # Ingests a set of plain integers, triggering full dynamic multi-file stitching inside the submodule
    high_digit_set = {10, 11, 12, 13}
    regression(targets=high_digit_set)

    # -------------------------------------------------------------------------
    # Stage 4: Extreme-Digit Compound Slice Regression (12 to 13 Digits)
    # -------------------------------------------------------------------------
    print("\n------------------------------------------------------------------------------")
    print(">>> [STAGE 4 / 4] Executing Compound Regression: 12 to 13 Digits Set...")
    print("------------------------------------------------------------------------------")
    # Isolating extreme boundary primes to monitor OLS asymptotic coefficient conservation bounds
    extreme_digit_set = {12, 13}
    regression(targets=extreme_digit_set)

    # -------------------------------------------------------------------------
    # Pipeline Summary Terminal Report
    # -------------------------------------------------------------------------
    total_elapsed = time.time() - total_start_time
    print("\n==============================================================================")
    print(f"🎉 MASTER REPRODUCTION SUCCESSFULLY COMPLETED! (Total runtime: {total_elapsed:.2f}s)")
    print("==============================================================================")
    print("👉 Summary of generated assets inside '../reports/reports_reproduced/':")
    print("   1. [LRT]        11 individual monotonicity reports (3 to 13 digits)")
    print("   2. [Regression] Global baseline complexity model report")
    print("   3. [Regression] Aggregated 10-13 digit compound model report")
    print("   4. [Regression] Aggregated 12-13 digit extreme compound model report")
    print("==============================================================================\n")

if __name__ == "__main__":
    # Default entry point fires up the comprehensive replication testbed suite
    reproduce()

    # Advanced custom analysis testing block (uncomment to test individual methods in terminal)
    # lrt(9, 2)
    # regression({12, 13})