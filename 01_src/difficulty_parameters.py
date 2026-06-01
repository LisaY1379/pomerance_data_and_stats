import pandas as pd
import math
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

    estimated_class_numbers = []
    for m in multiples:
        D = (p + 1 - m) ** 2 - 4 * p
        estimated_class_numbers.append(math.sqrt(abs(D)))

    if multiples:
        param_b_avg = sum(estimated_class_numbers) / len(estimated_class_numbers)
    else:
        param_b_avg = 0.0

    return param_a_count, round(param_b_avg, 2)


def process_prime_data(file_path):
    df = pd.read_csv(file_path)

    df[['param_a_count', 'param_b_avg']] = (
        df['prime']
        .apply(lambda x: pd.Series(calculate_ecpp_params(x)))
    )

    df.to_csv(file_path, index=False)
    print(f"Output saved to: {file_path}")


if __name__ == "__main__":
    current_dir = Path(__file__).resolve().parent
    project_root = current_dir.parent

    data_folder = current_dir

    if not data_folder.exists():
        print(f"Directory not found: {data_folder}")
    else:
        csv_files = list(data_folder.glob('*.csv'))

        if not csv_files:
            print(f"No CSV files found in: {data_folder}")

        for file_path in csv_files:
            print(f"Processing: {file_path.name} ...")
            try:
                process_prime_data(file_path)
            except Exception as e:
                print(f"Error processing {file_path.name}: {e}")