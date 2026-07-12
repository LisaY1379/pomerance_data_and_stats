import csv
import os
import sys

def calculate_class_numbers(input_path, output_path):
    if not os.path.exists(input_path):
        print(f"[Error] Temporary staging file not found at: {input_path}")
        sys.exit(1)

    with open(input_path, 'r', encoding='utf-8') as infile, \
         open(output_path, 'w', newline='', encoding='utf-8') as outfile:

        reader = csv.reader(infile)
        writer = csv.writer(outfile)

        # Parse and process header block safely
        header = next(reader, None)
        if header:
            if "class_number" not in header:
                header.append("class_number")
            writer.writerow(header)

        count = 0
        for row in reader:
            if not row:
                continue

            try:
                # Index coordinates mapped across column definitions:
                # row[0]=prime, row[1]=A, row[2]=x0, row[3]=trials, row[4]=param_a_count
                p = int(row[0])
                A = int(row[1])
                x0 = int(row[2])
            except ValueError:
                # Safe pass-through fallback for anomalous rows
                writer.writerow(row)
                continue

            try:
                # Construct base Montgomery elliptic model curve surface
                E_base = EllipticCurve(GF(p), [0, A, 0, 1, 0])

                # Check if x0 maps to a valid point on the curve or its quadratic twist
                rhs = (x0 ** 3 + A * x0 ** 2 + x0) % p
                if kronecker(rhs, p) >= 0:
                    E = E_base
                else:
                    E = E_base.quadratic_twist()

                t = E.trace_of_frobenius()
                D = t ** 2 - 4 * p

                # Directly compute the Class Number of the imaginary quadratic field using PARI
                h = pari(D).qfbclassno()

                row.append(str(int(h)))
                writer.writerow(row)

                count += 1
                if count % 100 == 0:
                    print(f"  [Sage] Processed {count} records...")

            except Exception as e:
                row.append("ERROR")
                writer.writerow(row)
                print(f"  [Sage] Error when processing p={p}, A={A}: {e}")

    print(f"  [Sage] Subprocess completed successfully. Injected {count} class numbers.")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("[Error] Missing command-line parameters for Sage runtime engine.")
        sys.exit(1)
    calculate_class_numbers(sys.argv[1], sys.argv[2])