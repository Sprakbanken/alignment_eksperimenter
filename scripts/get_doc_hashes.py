import argparse
from pathlib import Path

import pandas as pd


def get_doc_hashes(input_path, output_file):
    """
    Scripts that fetches doc hashes from already annotated documents.
    """
    doc_hashes = set()

    for file_path in input_path.glob("*.csv"):
        df = pd.read_csv(file_path)
        for _, row in df.iterrows():
            for col in df.columns:
                if col.startswith("doc_hash"):
                    value = row[col]
                    doc_hashes.add(str(value))

    with open(output_file, "w", encoding="utf-8") as outfile:
        for doc_hash in doc_hashes:
            outfile.write(f"{doc_hash}\n")


def get_args():
    parser = argparse.ArgumentParser(
        description="Get document hashes from annotated documents."
    )
    parser.add_argument(
        "--input_path",
        type=Path,
        required=True,
        help="Path to folder containing annotated documents.",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        required=True,
        help="Path to the output file to save document hashes.",
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    get_doc_hashes(args.input_path, args.output_file)
