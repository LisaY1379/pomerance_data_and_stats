import os
import csv

def create_retro_report(filepath):
    if not os.path.exists(filepath):
        print(f"Error: {filepath} not found!")
        return

    successful_proofs = 0
    total_as_tried = 0

    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get('prime') and row['prime'].strip():
                successful_proofs += 1
                total_as_tried += int(row['trials'])

    average_a_per_prime = total_as_tried / successful_proofs if successful_proofs > 0 else 0

    report_content = f"""=== TRIPLE GENERATION BENCHMARK ===
Source Primes: {filepath}
Hardware Processing Time: null
-----------------------------------
Successful Proofs: {successful_proofs}
Total A's Tried: {total_as_tried}
Average A's Tried per Prime: {average_a_per_prime:.2f}
-----------------------------------"""

    filename_without_ext = os.path.splitext(os.path.basename(filepath))[0]
    file_dir = os.path.dirname(os.path.abspath(filepath))

    report_dir = file_dir.replace(f"{os.sep}data{os.sep}", f"{os.sep}reports{os.sep}")
    if report_dir == file_dir:
        report_dir = file_dir.replace("data", "reports")

    os.makedirs(report_dir, exist_ok=True)

    report_path = os.path.join(report_dir, f"{filename_without_ext}_report.txt")

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_content)

    print(f"✅ Success! Created report at: {report_path}")
    print("\n--- Report Preview ---")
    print(report_content)

create_retro_report("../data/processed/triples_metrics_12to15digits_seed1778731660.csv")