import argparse
import json
import os
import re
import pandas as pd
from tqdm import tqdm


def is_year_dir(name: str) -> bool:
    return re.fullmatch(r"maalfrid_(\d{4})", name) is not None


def extract_year(dir_name: str) -> int:
    m = re.fullmatch(r"maalfrid_(\d{4})", dir_name)
    return int(m.group(1)) if m else -1


def parse_filename(fname: str) -> tuple[str, str, str]:
    """
    Expects filename like <domain>_<lang>_<type>.jsonl, e.g. '113.no_nob_pdf.jsonl'.
    Returns (domain, lang, filetype).
    """
    if not fname.endswith(".jsonl"):
        raise ValueError(f"Not a .jsonl file: {fname}")
    stem = fname[:-6]  # remove .jsonl
    parts = stem.split("_")
    if len(parts) < 3:
        raise ValueError(f"Invalid filename format: {fname}")
    ftype = parts[-1]
    lang = parts[-2]
    domain = "_".join(parts[:-2])
    return domain, lang, ftype


def find_grouped_files(data_dir: str) -> dict[str, list[tuple[int, str]]]:
    """
    Find all jsonl files in maalfrid_YYYY directories and group them by <domain>_<lang>.
    Returns: { domain_lang: [(year, filepath), ...] } sorted by year asc, filepath asc.
    """
    groups: dict[str, list[tuple[int, str]]] = {}
    for name in os.listdir(data_dir):
        if not is_year_dir(name):
            continue
        year = extract_year(name)
        year_path = os.path.join(data_dir, name)
        if not os.path.isdir(year_path):
            continue
        for entry in os.listdir(year_path):
            fp = os.path.join(year_path, entry)
            if not os.path.isfile(fp):
                continue
            if not entry.endswith(".jsonl"):
                continue
            try:
                domain, lang, _ = parse_filename(entry)
            except ValueError:
                continue
            key = f"{domain}_{lang}"
            groups.setdefault(key, []).append((year, fp))
    # Sort so that earlier years come first (so "keep last" keeps newest)
    for key in groups:
        groups[key].sort(key=lambda t: (t[0], t[1]))
    return groups




def build_group_df(year_filepath_pairs: list[tuple[int, str]], domain_lang: str = "") -> pd.DataFrame:
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
            rows = read_jsonl(fp)
        except Exception as e:
            tqdm.write(f"[WARNING] Could not read file: {fp} ({e})")
            continue
        if not rows:
            continue

        df = pd.DataFrame(rows)
        frames.append(df)

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
        type=str,
        default=os.path.join(os.path.dirname(os.path.dirname(__file__)), "data"),
        help="Root directory containing maalfrid_YYYY dirs (default: ./data)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "data", "maalfrid_superset"
        ),
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
    data_dir = args.data_dir
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    groups = find_grouped_files(data_dir)

    # Find which groups are already done
    completed = {
        os.path.splitext(fn)[0]
        for fn in os.listdir(output_dir)
        if fn.endswith(".jsonl") and os.path.isfile(os.path.join(output_dir, fn))
    }

    exclude = set(args.exclude_domains or [])



    keys_to_process = sorted(
        k for k in groups.keys()
        if domain_from_key(k) not in exclude and k not in completed
    )

    print(f"Found {len(groups)} groups (domain+language).")
    print(f"Already done: {len(completed)}. Remaining: {len(keys_to_process)}.")

    for key in tqdm(
        keys_to_process,
        desc="Building superset",
        unit="group",
        dynamic_ncols=True,
    ):
        year_filepath_pairs = groups[key]
        tqdm.write(f"\nProcessing: {key} ({len(year_filepath_pairs)} files)")

        df = build_group_df(year_filepath_pairs, domain_lang=key)

        if df.empty:
            out_path = os.path.join(output_dir, f"{key}.jsonl")
            with open(out_path, "w", encoding="utf-8"):
                pass
            tqdm.write("  Empty group, wrote empty file.")
            continue

        rows_before = len(df)
        df = dedupe_keep_last(df)
        rows_after = len(df)

        tqdm.write(
            f"  Rows before dedupe: {rows_before}, after: {rows_after} (removed {rows_before - rows_after})"
        )

        out_path = os.path.join(output_dir, f"{key}.jsonl")
        write_jsonl(out_path, df)

    print(f"\nDone! Superset written to: {output_dir}")


if __name__ == "__main__":
    main()
