import numpy as np
from pathlib import Path


def analyze_p_values(target_digits):
    current_dir = Path(__file__).resolve().parent

    if current_dir.name == '01_src' or current_dir.name == 'src':
        project_root = current_dir.parent
    else:
        project_root = current_dir

    reports_dir = project_root / 'reports' / 'lrt_reports' / '10trials' / f'{target_digits}digits'

    if not reports_dir.exists():
        print(f"❌ Error: Cannot find target directory -> {reports_dir}")
        return

    p_values = []

    all_txt_files = list(reports_dir.glob('*_report.txt'))

    txt_files = [f for f in all_txt_files if not f.name.startswith("PValue_Global_Analysis")]

    if not txt_files:
        print(f"⚠️ No _report.txt files in {reports_dir}")
        return

    print(f"🔍 Scanning through {len(txt_files)} reports")

    for file_path in txt_files:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if "Asymptotic P-Value" in line:
                    try:
                        p_val_str = line.split(':')[1].strip()
                        p_val = float(p_val_str)
                        p_values.append(p_val)
                    except Exception as e:
                        print(f"⚠️ Failed processing {file_path.name}: {e}")
                    break

    if p_values:
        avg_p = np.mean(p_values)
        median_p = np.median(p_values)
        min_p = np.min(p_values)
        max_p = np.max(p_values)
        reject_count = sum(1 for p in p_values if p < 0.05)
        global_rate = (reject_count / len(p_values)) * 100

        report_lines = [
            "=" * 50,
            f"📊 {target_digits} digits P-Value Analysis",
            "=" * 50,
            f"Total Number of Analysis Reports: {len(p_values)}",
            f"P-Value mean : {avg_p:.6f}",
            f"P-Value median : {median_p:.6f}",
            f"P-Value minimum : {min_p:.6e}",
            f"P-Value maximum : {max_p:.6f}",
            "-" * 50,
            f"Total number of rejection {reject_count}",
            f"Rate of rejection: {global_rate:.2f}%",
            "=" * 50
        ]
        report_content = "\n".join(report_lines)

        print(report_content)

        output_filename = f"PValue_Global_Analysis_{target_digits}digits.txt"
        output_file_path = reports_dir / output_filename

        with open(output_file_path, 'w', encoding='utf-8') as out_f:
            out_f.write(report_content)

        print(f"💾 Saved report to: \n{output_file_path}\n")
    else:
        print("⚠️ Scanning complete, but no valid p-values.")

if __name__ == "__main__":
    analyze_p_values(target_digits=11)