import argparse
import logging
import re
import time
from collections import defaultdict
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from align_documents.utils import setup_logging

logger = logging.getLogger(__name__)


def is_year_dir(name: str) -> bool:
    return re.fullmatch(r"maalfrid_(\d{4})", name) is not None


def extract_year(dir_name: str) -> int:
    m = re.fullmatch(r"maalfrid_(\d{4})", dir_name)
    return int(m.group(1))


def parse_filename(fname: Path) -> tuple[str, str, str]:
    """
    Expects filename like <domain>_<lang>_<type>.jsonl, e.g. '113.no_nob_pdf.jsonl'.
    Returns (domain, lang, filetype).
    """
    stem = fname.stem
    parts = stem.split("_")
    ftype = parts[-1]
    lang = parts[-2]
    domain = "_".join(parts[:-2])
    return domain, lang, ftype


def find_grouped_files(data_dir: Path) -> dict[str, dict[int], Path]:
    """
    Find all jsonl files in maalfrid_YYYY directories and group them by <domain>_<lang>.
    Returns: { domain_lang: { year: [filepath, ...]  }
    """
    groups = defaultdict(lambda: defaultdict(list))
    for sub_dir in sorted(data_dir.iterdir()):
        if not sub_dir.is_dir():
            continue
        if not is_year_dir(sub_dir.name):
            continue

        logger.info("Extracting domain and language info from %s", sub_dir)
        year = extract_year(sub_dir.name)

        for fp in sorted(sub_dir.glob("*.jsonl")):
            domain, lang, _ = parse_filename(fp)
            groups[f"{domain}_{lang}"][year].append(fp)

    return groups


def add_rows_if_missing(df: pd.DataFrame, other_df: pd.DataFrame) -> pd.DataFrame:
    """Only add rows from other_df if they dont already exist in df"""
    if df.empty:
        df_to_keep = other_df
    else:
        url_is_new = ~other_df.url.isin(df.url)
        doc_hash_is_new = ~other_df.doc_hash.isin(df.doc_hash)
        keep = url_is_new & doc_hash_is_new
        df_to_keep = pd.concat([df, other_df[keep]], ignore_index=True)

    return df_to_keep.drop_duplicates(subset=["url", "doc_hash"], ignore_index=True)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create a deduplicated superset of Målfrid dataset across years (per domain+language)."
    )
    parser.add_argument(
        "--data_dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "data",
        help="Root directory containing maalfrid_YYYY dirs (default: %(default)s)",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent
        / "data"
        / "maalfrid_superset_alt",
        help="Output directory for superset (default: %(default)s)",
    )
    parser.add_argument(
        "--exclude_domains",
        type=str,
        nargs="*",
        default=[],
        help="List of domains to exclude (e.g. 'regjeringen.no').",
    )
    parser.add_argument(
        "--log_level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set the logging level",
    )
    return parser.parse_args()


def domain_from_key(k: str) -> str:
    return k.rsplit("_", 1)[0]


def main():
    args = parse_args()
    setup_logging("make_malfrid_superset", args.log_level)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    groups = find_grouped_files(args.data_dir)
    # logger.debug(groups)

    # Find which groups are already done
    completed = {
        p.stem
        for p in args.output_dir.iterdir()
        if p.suffix == ".jsonl" and p.is_file()
    }

    exclude = set(args.exclude_domains or [])

    keys_to_process = sorted(
        k
        for k in groups.keys()
        if domain_from_key(k) not in exclude and k not in completed
    )

    logger.info("Found %d groups (domain+language).", len(groups))
    logger.info(
        "Already done: %d. Remaining: %d.", len(completed), len(keys_to_process)
    )

    t0 = time.perf_counter()
    for domain_lang in tqdm(
        keys_to_process,
        desc="Building superset",
        unit="group",
        dynamic_ncols=True,
    ):
        years_filepaths = groups[domain_lang]
        logger.info(
            "Processing: %s (files from %d years)", domain_lang, len(years_filepaths)
        )

        logger.debug(years_filepaths)
        years = sorted(years_filepaths, reverse=True)

        out_path = args.output_dir / f"{domain_lang}.jsonl"

        df = pd.DataFrame()
        for year in years:
            files = years_filepaths[year]
            logger.debug("Year %d, files: %s", year, files)
            rows_before_year = len(df)

            for fp in files:
                logger.debug("Reading %s", fp)
                rows_before = len(df)
                new_df = pd.read_json(fp, lines=True, orient="records")
                if new_df.empty:
                    logger.debug("Empty file: %s", fp)
                    continue
                df = add_rows_if_missing(df, other_df=new_df)
                rows_after = len(df)
                logger.debug("Added %s rows to df", rows_after - rows_before)

            rows_after_year = len(df)
            logger.info(
                "Rows before: %d, after: %d (added %d)",
                rows_before_year,
                rows_after_year,
                rows_after_year - rows_before_year,
            )

        df.to_json(out_path, orient="records", lines=True)

    elapsed = time.perf_counter() - t0
    logger.info("Loop completed in %.2f seconds.", elapsed)
    logger.info("Done! Superset written to: %s", args.output_dir)


if __name__ == "__main__":
    main()
