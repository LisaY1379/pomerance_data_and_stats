import pandas as pd
import random
import csv


def process_and_sample_prefixes(input_file, output_txt, output_csv, sample_size=10):
    """
    Reads a prefix file (p, A), counts A's, calculates exact probability,
    and samples primes from 3, 4, and 5-digit ranges.
    """
    print(f"Reading data from {input_file}...")

    prime_a_counts = {}
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            for row in reader:
                if not row or len(row) < 2:
                    continue
                try:
                    p = int(row[0])
                    prime_a_counts[p] = prime_a_counts.get(p, 0) + 1
                except ValueError:
                    continue
    except FileNotFoundError:
        print(f"Error: Could not find the input file '{input_file}'.")
        return

    if not prime_a_counts:
        print("Error: No valid data found in the input file.")
        return

    primes_by_digits = {3: [], 4: [], 5: []}

    for p, a_count in prime_a_counts.items():
        digits = len(str(p))
        if 3 <= digits <= 5:
            probability = a_count / p
            primes_by_digits[digits].append({
                'prime': p,
                'a_count': a_count,
                'exact_probability': probability
            })

    sampled_data = []

    for digits in [3, 4, 5]:
        pool = primes_by_digits[digits]
        actual_sample_size = min(sample_size, len(pool))

        if actual_sample_size > 0:
            sampled_data.extend(random.sample(pool, actual_sample_size))
            print(f"Sampled {actual_sample_size} primes with {digits} digits.")
        else:
            print(f"Warning: No primes found with {digits} digits in the data.")

    if not sampled_data:
        print("Error: No primes sampled in the 3-5 digit range.")
        return

    sampled_data.sort(key=lambda x: x['prime'])

    print("Generating output files...")

    with open(output_txt, 'w', encoding='utf-8') as f:
        for item in sampled_data:
            f.write(f"{item['prime']}\n")

    df = pd.DataFrame(sampled_data)
    df = df[['prime', 'a_count', 'exact_probability']]
    df.to_csv(output_csv, index=False)

    print(f"[Success] Generated TXT input list: {output_txt}")
    print(f"[Success] Generated CSV ground truth: {output_csv}")


if __name__ == "__main__":
    INPUT_FILE = "../data/final_data/pp16A.txt"

    OUTPUT_TXT = "sampled_primes_input.txt"

    OUTPUT_CSV = "sampled_primes_ground_truth.csv"

    SAMPLE_SIZE_PER_DIGIT = 10

    process_and_sample_prefixes(
        input_file=INPUT_FILE,
        output_txt=OUTPUT_TXT,
        output_csv=OUTPUT_CSV,
        sample_size=SAMPLE_SIZE_PER_DIGIT
    )