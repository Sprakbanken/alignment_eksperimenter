import argparse
import csv
import sys


def divide_csv(input_file):
    csv.field_size_limit(sys.maxsize)

    with open(input_file, "r", encoding="utf-8") as f:
        reader = list(csv.reader(f))
        header, rows = reader[0], reader[1:]

    chunk_size = len(rows) // 3

    for i in range(3):
        with open(
            f"all_alignments_part_{i + 1}.csv", "w", encoding="utf-8", newline=""
        ) as out:
            writer = csv.writer(out)
            writer.writerow(header)
            writer.writerows(rows[i * chunk_size : (i + 1) * chunk_size])


def parse_args():
    parser = argparse.ArgumentParser(
        description="Divide a CSV file into three equal parts."
    )
    parser.add_argument(
        "--input_file",
        type=str,
        default="all_test_files.csv",
        help="Path to the input CSV file.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    input_file = args.input_file
    divide_csv(input_file)
