import argparse
import glob
import hashlib
import os
from typing import Iterable

import pandas as pd
import tqdm


def compute_doc_hash(fulltext) -> str:
    """
    Compute hash over the newline-joined fulltext.
    """
    if isinstance(fulltext, list):
        text = "\n".join(fulltext)
    elif isinstance(fulltext, str):
        text = fulltext
    else:
        # Fallback to string representation
        text = str(fulltext)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def add_hash_column(
    df: pd.DataFrame, column_name: str = "fulltext", hash_column_name: str = "doc_hash"
) -> pd.DataFrame:
    """
    Adds a hash column based on the `fulltext` column.
    """
    if column_name in df.columns:
        df[hash_column_name] = df[column_name].apply(compute_doc_hash)
    else:
        print("No fulltext column found.")
    return df


def iter_jsonl_files(paths: Iterable[str]) -> list[str]:
    files: list[str] = []
    for p in paths:
        if os.path.isdir(p):
            files.extend(glob.glob(os.path.join(p, "**", "*.jsonl"), recursive=True))
        elif os.path.isfile(p) and p.endswith(".jsonl"):
            files.append(p)
    return files


def process_files(input_paths: list[str], output_dir: str | None) -> None:
    files = iter_jsonl_files(input_paths)
    if not files:
        raise ValueError("No JSONL files found in the given paths.")

    os.makedirs(output_dir, exist_ok=True) if output_dir else None

    for fp in tqdm.tqdm(files, desc="Processing files"):
        df = pd.read_json(fp, lines=True)
        df = add_hash_column(df, column_name="fulltext", hash_column_name="doc_hash")
        out_fp = os.path.join(output_dir, os.path.basename(fp)) if output_dir else fp
        df.to_json(out_fp, orient="records", lines=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="doc_hash based on fulltext for JSONL files."
    )
    parser.add_argument(
        "--input_paths",
        nargs="+",
        required=True,
        help="Files or directories containing .jsonl.",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Optional output directory. If omitted, files are modified in place.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    process_files(args.input_paths, args.output_dir)


if __name__ == "__main__":
    main()
