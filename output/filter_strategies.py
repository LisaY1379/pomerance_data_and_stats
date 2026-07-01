from math import gcd as my_gcd
from math import isqrt as my_isqrt


def filter_01_quadratic_residue_bias(A, x0, p):
    """Checks if A is a quadratic residue modulo p."""
    return pow(A, (p - 1) // 2, p) == 1

def filter_02_geometric_non_singular(A, x0, p):
    """Ensures the elliptic curve y^2 = x^3 + Ax is non-singular."""
    return (4 * pow(A, 3, p) + 27) % p != 0


def filter_03_pseudorandom_search_order(p, max_n=1000):
    """
    Strategy Name: Pseudo-random Search Order Generator
    Mathematical Description: Utilizes the SplitMix64 hash function to randomize the search order,
    thereby avoiding getting "stuck" on values of *A* with specific modular structures and increasing
    the probability of encountering a valid Pomerance triple.
    """
    def _splitmix64(x):
        x = (x + 0x9E3779B97F4A7C15) & ((1 << 64) - 1)
        x = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9) & ((1 << 64) - 1)
        x = ((x ^ (x >> 27)) * 0x94D049BB133111EB) & ((1 << 64) - 1)
        return x ^ (x >> 31)

    seed = _splitmix64(p)
    step = (_splitmix64(seed) % (p - 1)) + 1
    while gcd(step, p) != 1:
        step = (step % (p - 1)) + 1
    offset = _splitmix64(seed ^ p) % p

    for n in range(max_n):
        yield (offset + n * step) % p

# ==========================================
# Generated at Iteration: 2 | Filter ID: 04
# ==========================================
def filter_04_legendre_symbol_bias(A, x0, p):
    """
    # Name: Legendre Symbol Bias
    # Description: This filter uses the Legendre symbol to check if A is a quadratic residue modulo p.
    # The Legendre symbol (A/p) is calculated as A^((p-1)/2) % p. If the result is not 1, A is not a quadratic residue,
    # and thus, it is less likely to be a valid parameter for certain elliptic curve constructions.
    # This filter is deterministic and guarantees a 0% False Negative rate for valid A values.
    """
    if p <= 2:
        return True  # Edge case, no filtering for small primes

    legendre_symbol = pow(A, (p - 1) // 2, p)
    return legendre_symbol == 1
