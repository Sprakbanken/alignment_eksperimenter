import argparse
import logging
import re
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


def parse_filename(fname: str) -> tuple[str, str, str]:
    """
    Expects filename like <domain>_<lang>_<type>.jsonl, e.g. '113.no_nob_pdf.jsonl'.
    Returns (domain, lang, filetype).
    """
    if not fname.endswith(".jsonl"):
        raise ValueError(f"Not a .jsonl file: {fname}")
    stem = fname[:-6]  # remove .jsonl
    parts = stem.split("_")
    ftype = parts[-1]
    lang = parts[-2]
    domain = "_".join(parts[:-2])
    return domain, lang, ftype


def find_grouped_files(data_dir: Path) -> dict[str, list[tuple[int, Path]]]:
    """
    Find all jsonl files in maalfrid_YYYY directories and group them by <domain>_<lang>.
    Returns: { domain_lang: [(year, filepath), ...] } sorted by year asc, filepath asc.
    """
    groups: dict[str, list[tuple[int, Path]]] = {}
    for child in data_dir.iterdir():
        if not is_year_dir(child.name):
            continue
        year = extract_year(child.name)
        if not child.is_dir():
            continue
        for entry in child.iterdir():
            if not entry.is_file():
                continue
            try:
                domain, lang, _ = parse_filename(entry.name)
            except ValueError:
                continue
            key = f"{domain}_{lang}"
            groups.setdefault(key, []).append((year, entry))
    # Sort so that earlier years come first (so "keep last" keeps newest)
    for key in groups:
        groups[key].sort(key=lambda t: (t[0], t[1]))
    return groups


def build_group_df(year_filepath_pairs: list[tuple[int, Path]]) -> pd.DataFrame:
    """
    year_filepath_pairs: [(year, filepath), ...]
    Returns a DataFrame with all rows.
    """
    frames = []
    for year, fp in tqdm(
        year_filepath_pairs,
        leave=False,
        position=1,
        desc="Files",
        unit="file",
        dynamic_ncols=True,
    ):
        try:
            rows = pd.read_json(fp, lines=True)
        except Exception as e:
            logger.warning("Could not read file: %s (%s)", fp, e)
            continue
        if rows.empty:
            logger.info("Empty file: %s", fp)
            continue

        frames.append(rows)

    if not frames:
        return pd.DataFrame()

    return pd.concat(frames, ignore_index=True)


def dedupe_keep_last(df: pd.DataFrame) -> pd.DataFrame:
    """
    Deduplicate based on URL or doc-hash.
    Rows with same URL or same hash are considered duplicates.
    The newest (based on 'date' column) is kept.
    """

    # Convert date to string for sorting (some values may be int for some reason)
    df["_date_str"] = df["date"].astype(str)

    # Sort by date (oldest first) so "keep last" selects newest
    # ISO 8601 dates sort as strings (https://stackoverflow.com/questions/9576860/sort-iso-8601-dates-forward-or-backwards)
    df = df.sort_values("_date_str", kind="stable", na_position="first")
    url_col = "url" if "url" in df.columns else None
    hash_col = "doc_hash" if "doc_hash" in df.columns else None
    # Deduplicate on url
    if url_col is not None:
        has_url = df[url_col].notna() & (df[url_col] != "")

        df_with_url = df[has_url].copy()
        df_without_url = df[~has_url].copy()

        df_with_url = df_with_url.drop_duplicates(subset=[url_col], keep="last")

        df = pd.concat([df_with_url, df_without_url], ignore_index=True)
        df = df.sort_values("_date_str", kind="stable", na_position="first")

    # Deduplicate on hash
    if hash_col is not None:
        has_hash = df[hash_col].notna() & (df[hash_col] != "")

        df_with_hash = df[has_hash].copy()
        df_without_hash = df[~has_hash].copy()

        df_with_hash = df_with_hash.drop_duplicates(subset=[hash_col], keep="last")

        df = pd.concat([df_with_hash, df_without_hash], ignore_index=True)

    # Remove helper column
    df = df.drop(columns=["_date_str"], errors="ignore")

    return df


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
    return parser.parse_args()


def domain_from_key(k: str) -> str:
    return k.rsplit("_", 1)[0]


def main():
    args = parse_args()
    data_dir: Path = args.data_dir
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    groups = find_grouped_files(data_dir)

    # Find which groups are already done
    completed = {
        p.stem for p in output_dir.iterdir() if p.suffix == ".jsonl" and p.is_file()
    }

    exclude = set(args.exclude_domains or [])

    keys_to_process = sorted(
        k
        for k in groups.keys()
        if domain_from_key(k) not in exclude and k not in completed
    )

    logger.info("Found %s groups (domain+language).", len(groups))
    logger.info(
        "Already done: %s. Remaining: %s.", len(completed), len(keys_to_process)
    )

    for key in tqdm(
        keys_to_process,
        desc="Building superset",
        unit="group",
        dynamic_ncols=True,
    ):
        year_filepath_pairs = groups[key]
        logger.info("Processing: %s (%s files)", key, len(year_filepath_pairs))

        df = build_group_df(year_filepath_pairs)

        if df.empty:
            out_path = output_dir / f"{key}.jsonl"
            out_path.write_text("")
            logger.info("Empty group, wrote empty file.")
            continue

        rows_before = len(df)
        df = dedupe_keep_last(df)
        rows_after = len(df)

        logger.info(
            "Rows before dedupe: %s, after: %s (removed %s)",
            rows_before,
            rows_after,
            rows_before - rows_after,
        )

        out_path = output_dir / f"{key}.jsonl"
        df.to_json(out_path, orient="records", lines=True)

    logger.info("Done! Superset written to: %s", output_dir)


if __name__ == "__main__":
    setup_logging(source_script="make_målfrid_superset", log_level="INFO")
    main()
