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


def dedupe_keep_last(df: pd.DataFrame) -> pd.DataFrame:
    """
    Deduplicate based on URL or doc-hash.
    Rows with same URL or same hash are considered duplicates.
    The newest (based on 'date' column) is kept.
    """

    # Convert date to string for sorting (some values may be int for some reason)
    df["date"] = df["date"].astype(str)

    # Sort by date (oldest first) so "keep last" selects newest
    # ISO 8601 dates sort as strings (https://stackoverflow.com/questions/9576860/sort-iso-8601-dates-forward-or-backwards)
    df = df.sort_values("date", kind="stable", na_position="first")

    url_col = "url"
    hash_col = "doc_hash"

    # Deduplicate on url
    has_url = df[url_col].notna() & (df[url_col] != "")

    df_with_url = df[has_url].copy()
    df_without_url = df[~has_url].copy()

    df_with_url = df_with_url.drop_duplicates(subset=[url_col], keep="last")

    df = pd.concat([df_with_url, df_without_url], ignore_index=True)
    df = df.sort_values("date", kind="stable", na_position="first")

    # Deduplicate on hash
    has_hash = df[hash_col].notna() & (df[hash_col] != "")
    assert all(has_hash)
    return df.drop_duplicates(subset=[hash_col], keep="last").reset_index(drop=True)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create a deduplicated superset of Målfrid dataset across years (per domain+language)."
    )
    parser.add_argument(
        "--data_dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "data",
        help="Root directory containing maalfrid_YYYY dirs (default: ./data)",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "data" / "maalfrid_superset",
        help="Output directory for superset (default: ./data/maalfrid_superset)",
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

        out_path = args.output_dir / f"{domain_lang}.jsonl"

        logger.debug("Reading dfs")
        dfs = [
            pd.read_json(fp, lines=True, orient="records")
            for year, fps in years_filepaths.items()
            for fp in fps
        ]
        df = pd.concat([e for e in dfs if not e.empty])
        rows_before = len(df)
        df = dedupe_keep_last(df=df)
        rows_after = len(df)
        logger.info(
            "Rows before: %d, after: %d (removed %d)",
            rows_before,
            rows_after,
            rows_before - rows_after,
        )

        df.to_json(out_path, orient="records", lines=True)

    elapsed = time.perf_counter() - t0
    logger.info("Loop completed in %.2f seconds.", elapsed)
    logger.info("Done! Superset written to: %s", args.output_dir)


if __name__ == "__main__":
    main()
