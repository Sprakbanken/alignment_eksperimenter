import argparse
import logging
import re
from collections import defaultdict
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from align_documents.utils import setup_logging
from align_documents.utils.dataframe import dedupe_keep_last

logger = logging.getLogger(__name__)


def extract_year(dir_name: str) -> int | None:
    m = re.fullmatch(r"maalfrid_(\d{4})", dir_name)
    return int(m.group(1)) if m else None


def find_grouped_files(data_dir: Path) -> dict[tuple[str, str], list[Path]]:
    """
    Find all jsonl files in maalfrid_YYYY directories and group them by (domain, lang).
    Returns: { (domain, lang): [filepath1, filepath2, ...] }
    """
    groups: dict[tuple[str, str], list[tuple[int, Path]]] = defaultdict(list)
    for year_dir in data_dir.iterdir():
        if not year_dir.is_dir():
            continue
        year = extract_year(year_dir.name)
        if year is None:
            continue
        for entry in year_dir.glob("*.jsonl"):
            domain, lang, _ftype = entry.stem.split("_")
            groups[(domain, lang)].append(entry)
    return groups


def build_df(jsonl_files: list[Path]) -> pd.DataFrame:
    """Returns a DataFrame with all rows of the jsonl files."""
    frames = []
    for filepath in jsonl_files:
        try:
            rows = pd.read_json(filepath, lines=True)
        except Exception:
            logger.exception("Could not read file: %s", filepath)
            continue
        if rows.empty:
            logger.info("Empty file: %s", filepath)
            continue

        frames.append(rows)

    if not frames:
        return pd.DataFrame()

    return pd.concat(frames, ignore_index=True)


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
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level (default: INFO).",
    )
    return parser.parse_args()


def main(args):
    data_dir: Path = args.data_dir
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    groups = find_grouped_files(data_dir)

    # Find which groups are already done
    completed = {p.stem for p in output_dir.glob("*jsonl")}

    domains_to_exclude = set(args.exclude_domains or [])

    domain_langs_to_process = sorted(
        (domain, lang)
        for (domain, lang) in groups
        if domain not in domains_to_exclude and f"{domain}_{lang}" not in completed
    )

    logger.info("Found %s groups (domain+language).", len(groups))
    logger.info(
        "Already done: %s. Remaining: %s.", len(completed), len(domain_langs_to_process)
    )

    for domain, lang in tqdm(
        domain_langs_to_process,
        desc="Building superset",
        unit="group",
        dynamic_ncols=True,
    ):
        filepaths = groups[(domain, lang)]

        logger.info(
            "Processing domain: %s lang: %s (%s files)",
            domain,
            lang,
            len(filepaths),
        )

        df = build_df(filepaths)

        if df.empty:
            logger.info("Empty group, skipping.")
            continue

        rows_before = len(df)
        df = dedupe_keep_last(df)
        rows_after = len(df)

        logger.debug(
            "Rows before dedupe: %s, after: %s (removed %s)",
            rows_before,
            rows_after,
            rows_before - rows_after,
        )

        out_path = output_dir / f"{domain}_{lang}.jsonl"
        df.to_json(out_path, orient="records", lines=True)

    logger.info("Done! Superset written to: %s", output_dir)


if __name__ == "__main__":
    args = parse_args()
    setup_logging(source_script="make_målfrid_superset", log_level=args.log_level)
    main(args)
