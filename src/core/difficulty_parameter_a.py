import pandas as pd
import math
import argparse
from pathlib import Path

def calculate_ecpp_params(p):
    q = math.isqrt(p)
    target = q + 1 + 2 * math.sqrt(q)

    k = math.floor(math.log2(target)) + 1 if target > 0 else 1
    two_k = 2 ** k

    hasse_low = p + 1 - 2 * math.sqrt(p)
    hasse_high = p + 1 + 2 * math.sqrt(p)

    m_start = math.ceil(hasse_low / two_k) * two_k
    m_end = math.floor(hasse_high / two_k) * two_k

    multiples = []
    curr = m_start
    while curr <= m_end:
        multiples.append(curr)
        curr += two_k

    param_a_count = len(multiples)

    return param_a_count

def process_prime_data(input_file, output_file=None):
    df = pd.read_csv(input_file)

    if 'prime' not in df.columns:
        raise ValueError("The input CSV must contain a 'prime' column.")

    df['param_a_count'] = df['prime'].apply(calculate_ecpp_params)

    if output_file is None:
        input_path = Path(input_file)
        output_file = input_path.with_name(f"{input_path.stem}_processed{input_path.suffix}")

    df.to_csv(output_file, index=False)
    print(f"Output successfully saved to: {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate ECPP param_a_count for a specific CSV file.")
    parser.add_argument("input_csv", help="Path to the input CSV file.")
    parser.add_argument("-o", "--output_csv", help="Path for the output CSV file (optional).", default=None)

    args = parser.parse_args()
    input_path = Path(args.input_csv)

    if not input_path.exists():
        print(f"Error: The file '{input_path}' does not exist.")
    elif not input_path.is_file() or input_path.suffix.lower() != '.csv':
        print(f"Error: '{input_path}' is not a valid CSV file.")
    else:
        print(f"Processing: {input_path.name} ...")
        try:
            process_prime_data(args.input_csv, args.output_csv)
        except Exception as e:
            print(f"Error processing {input_path.name}: {e}")