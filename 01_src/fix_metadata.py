import os
import json


def create_retro_metadata(filepath, min_digits, max_digits, seed):
    if not os.path.exists(filepath):
        print(f"Error: {filepath} not found!")
        return

    with open(filepath, 'r') as f:
        batch_size = sum(1 for line in f if line.strip())

    metadata = {
        "batch_size": batch_size,
        "min_digits": min_digits,
        "max_digits": max_digits,
        "seed_used": seed,
        "total_generation_time_seconds": None,
        "average_time_per_prime_seconds": None,
        "note": "Retroactively created metadata. Exact line count verified."
    }

    meta_path = filepath.replace(".txt", "_deleted_duplicates_meta.json")
    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=4)

    print(f"✅ Success! Created metadata for {os.path.basename(filepath)}: verified {batch_size} primes.")

create_retro_metadata("../data/raw/primes_12to15digits_seed1778731660_full.txt", 12, 15, 1778731660)