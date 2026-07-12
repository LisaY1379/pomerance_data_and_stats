# Pomerance Data and Stats

This repository delivers a systematic benchmark investigating the following two key areas in the context of Pomerance primality proofs, regarding the [DANGER3 data challenge](https://github.com/AndrewVSutherland/DANGER3):
1. **Search Volatility**: How the number of trials required to find a Pomerance triple fluctuates across different prime scales.
2. **Correlation Analysis**: The key structural factors affecting search difficulty and their statistical correlations.

## Key Findings

> **TL;DR:** While the search difficulty naturally scales with the size of the prime, it exhibits **massive local volatility**. However, this apparent randomness is not blind—it is strictly governed by a **log-linear relationship** with the curve's algebraic parameters $A$ and $B$.

### 1. Search Volatility (Isotonic LRT)
Given that our algorithm randomly samples the parameter $A$, the number of trials required to find a valid Pomerance triple naturally follows a **Geometric Distribution**. We systematically investigated primes spanning from 3 to 13 digits. 

To rigorously test whether search difficulty strictly scales with prime magnitude, we conducted an **Isotonic Likelihood Ratio Test (Isotonic LRT)** within each digit interval (from 3-digit up to 13-digit segments). 

* **The Statistical Setup**:
  * $\mathcal{H}_0$ (Monotonicity Hypothesis): Expected search trials are monotonically non-decreasing with respect to the primes within the same digit scale ($T_{p_1} \le T_{p_2} \le \dots \le T_{p_n}$ for sorted primes $p_i$).
  * $\mathcal{H}_1$ (Unconstrained Space): Expected trials can fluctuate freely across the prime spectrum.
* **The Empirical Result**:
  * Within every single digit group, the null hypothesis $\mathcal{H}_0$ was **rejected with a $P$-value approaching 0**. 
  * **Scientific Interpretation**: This indicates that while the search difficulty exhibits a clear upward trend *macroscopically* as primes grow across digit scales (Global Monotonicity), it exhibits high **local stochastic volatility** *microscopically*. The rigid non-decreasing assumption is violated frequently in local windows due to the unique properties of individual primes. Thus, finding a Pomerance triple is governed by a global trend but heavily driven by local randomness.

### 2. Correlation Analysis (Log-Log Regression)
To understand what structural factors govern the local stochastic volatility, we extracted two fundamental algebraic parameters for each prime and curve:
* **Parameter A (`param_a_count`)**: The number of $2^k$ multiples residing within the Hasse interval. (Categorical: 1, 2, 3, or 4).
* **Parameter B (`class_number`)**: The class number derived from the curve's trace of Frobenius.

We conducted multiple Log-Log Ordinary Least Squares (OLS) regressions across different digit scales using the model: $\log(\text{trials}) = \beta_0 + \beta_1 \log(\sqrt{p}) + \beta_A + \beta_B \log(B)$.

* **Deterministic Log-Linear Relationship**: On the global dataset ($N=39,573$), the model explained an overwhelming **$96.2\%$ of the variance** ($R^2 = 0.962$). This reveals that while the search process relies on random sampling, the *expected difficulty* is almost completely deterministic, governed tightly by a log-linear relationship with parameters $A$ and $B$.
* **The Hasse Multiplier Effect (Param A)**: Primes yielding more $2^k$ multiples are exponentially easier to solve. The categorical coefficients strictly followed a monotonically decreasing impact ($\beta_{A=4} < \beta_{A=3} < \beta_{A=2} < 0$).
* **Multicollinearity & The $\log(\sqrt{p})$ Invariant (Param B)**: A striking mathematical artifact emerges between the prime magnitude $p$ and the class number $B$. Theoretically, search complexity scales linearly with $\sqrt{p}$ (expected coefficient of $1$). Because $B$ is bounded by the Hasse interval (which also scales with $\sqrt{p}$), the two variables exhibit strong multicollinearity. Our empirical data perfectly captures this invariant: while the regression splits the weights, their coefficient sum strictly conserves the theoretical bound ($\beta_{\log(\sqrt{p})} + \beta_{\log(B)} \approx 1.286 - 0.292 \approx 1$).

## Reproduction

Our analysis workflow is entirely structured into an automated pipeline. You can reproduce the statistical findings and regenerate the analytical plots using either:
1. our precomputed dataset;
2. or by generating fresh search data from scratch.

### Prerequisites

Ensure you have the required Python packages: 
```bash
pip install numpy pandas statsmodels scikit-learn joblib cypari2 matplotlib
```
If you wish to generate original data, ensure 'gcc' or 'clang' is available for the C binary compilation.

### Option 1: One-Click Statistical Replication (Using Precomputed Data)

*Estimated runtime: 3min*

If you want to instantly replicate the isotonic LRT and the regression model using our precomputed database

1. Open this repository in PyCharm
2. Navigate to src/run_analysis.py and open it
3. Make sure this is in the main function:
```python
if __name__ == "__main__":
    reproduce() #the function reproducing all results
```
4. Click the green "Run" button in the top right corner

### Option 2: Generate Original Data
*Estimated runtime: Variable depending on hardware core limits, target digit and sample size*

To validate the entire structural stack—from parallel C-level sampling to feature extraction and report generation:

#### Step 1: Generate Data

1. Open src/pomerance_driver.py. 
2. Configure your desired target chunk size (e.g., digits=9, triple_amount_for_each_prime=100, number_of_primes=20). 
*Note: triple_amount_for_each_prime >= 20 is recommended for an accurate result.*
3. Click the green "Run" button

The new data will be in the form:`[digit_number]_digits_v[version_number] e.g: 3_digits_v2`

Please check out the version number of your data before sending them into the analytics.

#### Step 2: Analyze Original Data (Navigate to src/run_analysis.py)

* **To Validate LRT of Specific Data**:
  * Input the digit and version number of targeted data (if no version information is given, it will analyze the precomputed data of the given digit):
```python
#Example Usage
if __name__ == "__main__":
    lrt(3, 2) #3_digits_v2
    lrt(4, 3) #4_digits_v3
    lrt(3) #3_digits_precomputed
```
* **To Validate Regression of Specific Data**:
  * Input the datasets you wish to validate, each dataset is represented as a tuple `([digit_number],[version_number]) `
  
  `e.g: (3, 2) -> 3_digits_v2`
  * If no version number is given it will represent the precomputed data of the given digit `e.g: (3) -> 3_digits_precomputed`
```python
#Example Usage
if __name__ == "__main__":
    regression({(3, 2), (4, 3), (5, 2)})
    regression({(3), (6, 3)})
```

## Repository Layout

```text
├── src/                    
│   ├── run_analysis.py        # Entry point for reproduction and LRT/Regression analysis
│   ├── pomerance_driver.py    # Entry point for original data generation
│   ├── pomerance.c            # Search code from Fabian Ruehle with Claude Opus 4.6 (MIT)
│   └── core/                  # Backend modules for analytic & algebraic number theory compute 
│                             (include computing module for param_a_count, class_number, LRT and Regression engine)
├── data/                      # Storage for all precomputed and original data
├── reports/                  
│   ├── lrt/                   # Storage for all lrt reports and plots from precomputed data
│   ├── regression/            # Storage for all regression reports and plots from precomputed data
│   ├── reports_reproduced/    # [Generated at runtime] Storage for reproduced LRT & Regression reports
│   └── reports_original_data/ # [Generated at runtime] Storage for reports of original & customized data
├── .gitignore                 # Files excluded from version control
├── LICENSE                    # Open-source licensing terms
└── README.md                  # This file
```
