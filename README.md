# Pomerance Data and Stats

This repository delivers a systematic benchmark investigating the following two key areas in the context of Pomerance primality proofs, regarding the [DANGER3 data challenge](https://github.com/AndrewVSutherland/DANGER3):
1. **Search Volatility**: How the number of trials required to find a Pomerance triple fluctuates across different prime scales.
2. **Correlation Analysis**: The key structural factors affecting search difficulty and their statistical correlations.

## Description of Data
The dataset `data/` contains Pomerance triples for some primes ranging from 3 to 13 digits. 20 triples are generated for each prime.

### What is a Pomerance Triple?

A Pomerance proof verifies that an odd integer $p$ is prime via a Pomerance triple $(p, A, x_0)$ where $0 \le A, x_0 < p$ and $A \not\equiv \pm 2 \pmod p$. It demands that the projective point $(x_0 : 1)$ on the Montgomery curve $By^2 = x^3 + Ax^2 + x$ has an exact order of $2^k$, where $k$ is the least integer satisfying $2^k > q + 1 + 2\sqrt{q}$ for $q = \lfloor\sqrt{p}\rfloor$. Concretely, applying the curve's doubling law exactly $k$ times modulo $p$ must yield a $Z$-coordinate congruent to zero, whereas the $(k-1)$-th doubling must leave a $Z$-coordinate coprime to $p$.

* **`prime`, `A`, `x0`**: The Pomerance triple.

* **`trials`**: Number of As tried before finding a correct A.

* **Parameter A (`param_a_count`)**: The number of $2^k$ multiples residing within the Hasse interval. (Categorical: 1, 2, 3, or 4).

* **Parameter B (`class_number`)**: The class number derived from the curve's trace of Frobenius.

The dataset provides data for testing/discovering properties of Pomerance proofs and any statistical measurements.

## Key Findings

> **TL;DR:** While the search difficulty naturally scales with the size of the prime, it exhibits **massive local volatility**. However, this apparent randomness is not blind—it is strictly governed by a **log-linear relationship** with the curve's algebraic parameters $A$ and $B$.

### 1. Search Volatility (Isotonic LRT)
Given that our algorithm randomly samples the parameter $A$, the number of trials required to find a valid Pomerance triple naturally follows a **Geometric Distribution**. We systematically investigated primes spanning from 3 to 13 digits. 

To rigorously test whether search difficulty strictly scales with prime magnitude, we conducted an **Isotonic Likelihood Ratio Test (Isotonic LRT)** within each digit interval (from 3-digit up to 13-digit segments). 

* **The Statistical Setup**:
  * $\mathcal{H}\_{0}$ (Monotonicity Hypothesis): Expected search trials are monotonically non-decreasing with respect to the primes within the same digit scale ($T\_{p\_1} \le T\_{p\_2} \le \dots \le T\_{p\_n}$ for sorted primes $p_i$).
  * $\mathcal{H}_1$ (Unconstrained Space): Expected trials can fluctuate freely across the prime spectrum.
* **The Empirical Result**:
  * Within every single digit group, the null hypothesis $\mathcal{H}_0$ was **rejected with a $P$-value approaching 0**. 
  * **Scientific Interpretation**: This indicates that while the search difficulty exhibits a clear upward trend *macroscopically* as primes grow across digit scales (Global Monotonicity), it exhibits high **local stochastic volatility** *microscopically*. The rigid non-decreasing assumption is violated frequently in local windows due to the unique properties of individual primes. Thus, finding a Pomerance triple is governed by a global trend but heavily driven by local randomness.

## 2. Log-Log Regression Analysis (Final Model)

To evaluate how structural factors govern local stochastic volatility without multicollinearity artifacts, we orthogonalized `log(p)` against `log(B)` based on the theoretical Hasse interval constraint. The resulting residual **ε** isolates the unique variance of `log(p)` independent of class number `B`.

Our final log-log Ordinary Least Squares (OLS) model is specified as:


$$\log(\text{trials}) = \beta_0 + \beta_{\text{res}} \cdot \varepsilon + \beta_A + \beta_B^* \log(B)$$


On the global dataset (**N = 39,573**), the final model yields exceptional explanatory power (**R² = 0.962**), confirming that search difficulty is almost completely deterministic:

- **Deterministic Log-Linear Structure**: Despite random sampling in the search process, expected difficulty strictly follows a log-linear relationship with parameters A (`param_a_count`) and B (`class_number`).

- **The Hasse Multiplier Effect (Param A)**: Primes yielding more 2k multiples are exponentially easier to solve, as categorical coefficients follow a strict monotonic decrease ($\beta_{A=4} < \beta_{A=3} < \beta_{A=2} < 0$).

- **Class Number Invariant & Scaling (Param B)**: By eliminating collinear weight distortion, the class number coefficient $\beta_B^*$ converges neatly to **0.956** (≈ 1.0), perfectly aligning with the theoretical complexity bound. Meanwhile, $\beta_{\text{res}}$ = **1.287** independently captures the baseline stochastic scaling difficulty of the prime magnitude.

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
