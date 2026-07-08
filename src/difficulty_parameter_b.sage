import csv
import os
import sys


def get_absolute_path(relative_path_from_script):
    script_dir = os.path.dirname(os.path.abspath(__file__ if '__file__' in locals() else '.'))
    return os.path.normpath(os.path.join(script_dir, relative_path_from_script))


def calculate_class_numbers(input_path, output_path):

    if not os.path.exists(input_path):
        print(f"[Error] Input file not found, please make sure it exists at: {os.path.abspath(input_path)}")
        return

    print(f"Starting to read the file: {input_path}")

    with open(input_path, 'r', encoding='utf-8') as infile, \
         open(output_path, 'w', newline='', encoding='utf-8') as outfile:

        reader = csv.reader(infile)
        writer = csv.writer(outfile)

        count = 0

        for row in reader:
            if not row:
                continue

            try:
                p = int(row[0])
                A = int(row[1])
                x0 = int(row[2])
            except ValueError:
                if "class_number" not in row:
                    row.append("class_number")
                writer.writerow(row)
                continue

            try:
                E_base = EllipticCurve(GF(p), [0, A, 0, 1, 0])

                rhs = (x0 ** 3 + A * x0 ** 2 + x0) % p
                if kronecker(rhs, p) >= 0:
                    E = E_base
                else:
                    E = E_base.quadratic_twist()

                t = E.trace_of_frobenius()
                D = t ** 2 - 4 * p

                h = pari(D).qfbclassno()

                row.append(str(int(h)))
                writer.writerow(row)

                count += 1
                if count % 100 == 0:
                    print(f"Processed {count} records...")

            except Exception as e:
                row.append("ERROR")
                writer.writerow(row)
                print(f"Error when processing p={p}, A={A}: {e}")

    print(f"[Success] Processing complete. Total amount: {count} triples.")
    print(f"Results have been safely saved to: {output_path}")

if __name__ == "__main__":
     calculate_class_numbers(sys.argv[1], sys.argv[2])