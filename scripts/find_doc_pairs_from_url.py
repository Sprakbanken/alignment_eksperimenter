import argparse
import logging
import re
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
from rapidfuzz import fuzz

from align_documents.utils.config import get_config
from align_documents.utils import setup_logging

logger = logging.getLogger(__name__)


def normalize_url(url):
    return re.sub(r"/[a-z]{2}-[A-Z]{2}/", "/xx-XX/", url)


def is_valid_url_pair(url1, url2, threshold=97):
    return fuzz.ratio(normalize_url(url1), normalize_url(url2)) >= threshold


def ends_with_digit(string):
    match = re.search(r"(\d+)(?=\.$)", string)
    return bool(match) if match else False


def match_rows(
    lang_code_1_row, lang_code_2_rows, lang_code_1, lang_code_2
):
    matches = []
    lang_code_1_url = getattr(lang_code_1_row, "url")

    for lang_code_2_row in lang_code_2_rows:
        lang_code_2_url = getattr(lang_code_2_row, "url")
        if is_valid_url_pair(lang_code_2_url, lang_code_1_url):
            matches.append(
                {
                    f"{lang_code_1}_doc_hash": getattr(lang_code_1_row, "doc_hash"),
                    f"{lang_code_2}_doc_hash": getattr(lang_code_2_row, "doc_hash"),
                    f"{lang_code_1}_url": getattr(lang_code_1_row, "url"),
                    f"{lang_code_2}_url": getattr(lang_code_2_row, "url"),
                    f"{lang_code_1}_fulltext": getattr(lang_code_1_row, "fulltext"),
                    f"{lang_code_2}_fulltext": getattr(lang_code_2_row, "fulltext"),
                }
            )
    return matches


def contains_dates_or_many_numbers(url):
    return (
        bool(re.search(r"/\d{4}(/|$)", url))  # match years
        or bool(re.search(r"\d{6,}", url))  # match long sequences of digits
        or bool(re.search(r"/\d+/?$", url))  # match a number at the end (e.g 06/)
        or bool(
            re.search(r"/[^/]*\d+/?$", url)
        )  # match just a digit at the end (e.g 06)
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create document pairs based on URL similarity"
    )
    parser.add_argument(
        "--log_level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set the logging level",
    )

    parser.add_argument("--config_file", default="alignment_config.toml")
    args = parser.parse_args()
    setup_logging("make_pairs_based_on_url", args.log_level)
    config = get_config(config_file=args.config_file)

    source_p = config.data_dir
    lang_code_1 = config.languages[0]
    lang_code_2 = config.languages[1]
    output_dir = config.output_dir / "url_pairs"

    output_dir.mkdir(exist_ok=True, parents=True)

    file_group_regex = re.compile(rf"(?P<domain>.*?)_(?P<lang_code>{lang_code_1}|{lang_code_2})")

    domain_groups = defaultdict(dict)
    for file in source_p.iterdir():
        if match := file_group_regex.match(file.stem):
            domain = match.group('domain')
            lang_code = match.group('lang_code')
            domain_groups[domain][lang_code] = file
        else:
            logger.warning(
                f"File {file.name} does not match expected pattern and will be skipped."
            )

    # Filter out domains missing desired lang codes
    domain_groups_filtered = {
        domain: langs
        for domain, langs in domain_groups.items()
        if set([lang_code_1, lang_code_2]).issubset(langs.keys())
    }
    logger.info(f"Domain groups: {len(domain_groups_filtered)}")

    # Process each domain
    for domain, langs in domain_groups_filtered.items():
        logger.info(f"Processing domain: {domain}")

        df_lang_code_1 = pd.read_json(langs[lang_code_1], lines=True)
        df_lang_code_1 = df_lang_code_1[~df_lang_code_1['url'].apply(contains_dates_or_many_numbers)]

        df_lang_code_2 = pd.read_json(langs[lang_code_2], lines=True)
        df_lang_code_2 = df_lang_code_2[~df_lang_code_2['url'].apply(contains_dates_or_many_numbers)]

        df_lang_code_2_rows = list(df_lang_code_2.itertuples())
        all_matches = []

        with ThreadPoolExecutor() as executor:
            results = executor.map(
                lambda row: match_rows(
                    row, df_lang_code_2_rows, lang_code_1, lang_code_2
                ),
                df_lang_code_1.itertuples(),
            )
            for matched_rows in results:
                all_matches.extend(matched_rows)

        result_df = pd.DataFrame(all_matches)
        logger.info(f"Matches found for {domain}: {len(result_df)}")

        if len(result_df) > 0:
            result_df_filtered = result_df[
                ~result_df[f"{lang_code_1}_url"].apply(ends_with_digit)
                & ~result_df[f"{lang_code_2}_url"].apply(ends_with_digit)
            ]
        else:
            result_df_filtered = pd.DataFrame()

        (output_dir / f"{lang_code_1}_{lang_code_2}").mkdir(exist_ok=True, parents=True)

        if len(result_df_filtered) > 0:
            logger.info(
                f"Writing {len(result_df_filtered)} valid matches for {domain} to file."
            )
            result_df_filtered.to_csv(
                output_dir / f"{lang_code_1}_{lang_code_2}" / f"{domain}.csv",
                index=False,
            )
        else:
            logger.warning(
                f"No valid matches found for {domain} after filtering. Skipping file."
            )
