import argparse
import hashlib
from pathlib import Path
import logging

import pandas as pd
import tqdm

from align_documents.utils import setup_logging

logger = logging.getLogger(__name__)


def compute_doc_hash(fulltext: list[str]) -> str:
    """
    Compute hash over the newline-joined fulltext.
    """
    text = "\n".join(fulltext)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def create_files_with_dochash(
    input_dir: Path, output_dir: Path, overwrite: bool
) -> None:
    """Create doc_hash column based on fulltext column"""
    jsonl_files = list(input_dir.glob("*.jsonl"))
    if not jsonl_files:
        raise ValueError("No JSONL files found in the given paths.")

    output_dir.mkdir(exist_ok=True)

    for fp in tqdm.tqdm(jsonl_files, desc="Processing jsonl files"):
        out_fp = output_dir / fp.name

        if not overwrite and out_fp.exists() and out_fp != fp:
            logger.debug("%s already exists, skipping", out_fp)
            continue

        logger.debug("Reading %s", fp)
        df = pd.read_json(fp, lines=True)
        if "fulltext" not in df.columns:
            logger.warning("Column 'fulltext' not in jsonl file %s", fp)
            logger.debug("len(df): %d\t\tdf.columns: %s", len(df), df.columns)
            continue
        # add hash column
        df["doc_hash"] = df["fulltext"].apply(compute_doc_hash)

        out_fp = output_dir / fp.name
        logger.debug("Writing dataframe with doc_hash column to %s", out_fp)
        df.to_json(out_fp, orient="records", lines=True, index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="doc_hash based on fulltext for JSONL files."
    )
    parser.add_argument(
        "--input_dir",
        type=Path,
        default=Path("data/maalfrid_2021"),
        help="Directory containing .jsonl.",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        type=Path,
        help="Optional output directory. If omitted, files are modified in place.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="If flagged, will overwrite existing output files",
    )
    parser.add_argument(
        "--log_level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set the logging level",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    setup_logging("hash_documents", args.log_level)
    output_dir = args.output_dir if args.output_dir else args.input_dir
    create_files_with_dochash(args.input_dir, output_dir, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
