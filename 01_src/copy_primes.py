import os

def copy_primes_from_pure_txt(source_path, target_path, start_line):
    if not os.path.exists(source_path):
        print(f"Error: {source_path} does not exist！")
        return

    with open(source_path, 'r', encoding='utf-8') as f_in:
        lines = f_in.readlines()

    if not lines:
        print("Error: the file is empty！")
        return

    sliced_data = lines[start_line - 1:]

    if not sliced_data:
        print(f"Warning：line {start_line} out of range ({len(lines)} lines maximum)")
        return

    target_dir = os.path.dirname(target_path)
    if target_dir:
        os.makedirs(target_dir, exist_ok=True)

    with open(target_path, 'w', encoding='utf-8') as f_out:
        f_out.writelines(sliced_data)

    print(f"✅ Successfully copied")
    print(f"   Full length: {len(lines)}")
    print(
        f"   New file includes: lines {start_line} to {len(lines)}, {len(sliced_data)} numbers in total, saved to {target_path}")

source = "../data/raw/primes_12to15digits_seed1778731660.txt"
target = "../data/raw/primes_12to15digits_seed1778731660_from8813.txt"

copy_primes_from_pure_txt(source, target, start_line=8813)