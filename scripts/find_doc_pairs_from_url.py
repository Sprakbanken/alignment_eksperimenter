import argparse
import logging
import re
from collections import defaultdict
from typing import Callable
import pandas as pd
import numpy as np
from rapidfuzz import fuzz, process

from align_documents.utils.config import get_config
from align_documents.utils import setup_logging

logger = logging.getLogger(__name__)


def normalize_url(url):
    return re.sub(r"/[a-z]{2}-[A-Z]{2}/", "/xx-XX/", url)


def ends_with_digit(string):
    match = re.search(r"(\d+)(?=\.$)", string)
    return bool(match) if match else False


def contains_dates_or_many_numbers(url):
    return (
        bool(re.search(r"/\d{4}(/|$)", url))  # match years
        or bool(re.search(r"\d{6,}", url))  # match long sequences of digits
        or bool(re.search(r"/\d+/?$", url))  # match a number at the end (e.g 06/)
        or bool(
            re.search(r"/[^/]*\d+/?$", url)
        )  # match just a digit at the end (e.g 06)
    )


def compare_urls(
    df_lang_code_1: pd.DataFrame,
    df_lang_code_2: pd.DataFrame,
    scorer: Callable[..., float] = fuzz.ratio,
    score_cutoff: float = 97,
) -> pd.DataFrame:
    all_matches = []

    urls_normalized_1 = df_lang_code_1["url"].apply(normalize_url)
    urls_normalized_2 = df_lang_code_2["url"].apply(normalize_url)

    scores = process.cdist(
        urls_normalized_1, urls_normalized_2, scorer=scorer, score_cutoff=score_cutoff
    )
    indices = np.where(np.triu(scores, k=1))

    for res_1, res_2 in zip(*indices):
        all_matches.append(
            {
                f"{lang_code_1}_doc_hash": df_lang_code_1["doc_hash"].iloc[res_1],
                f"{lang_code_2}_doc_hash": df_lang_code_2["doc_hash"].iloc[res_2],
                f"{lang_code_1}_url": df_lang_code_1["url"].iloc[res_1],
                f"{lang_code_2}_url": df_lang_code_2["url"].iloc[res_2],
                f"{lang_code_1}_fulltext": df_lang_code_1["fulltext"].iloc[res_1],
                f"{lang_code_2}_fulltext": df_lang_code_2["fulltext"].iloc[res_2],
            }
        )

    return pd.DataFrame(all_matches)


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

    file_group_regex = re.compile(
        rf"(?P<domain>.*?)_(?P<lang_code>{lang_code_1}|{lang_code_2})"
    )

    logger.info("Collecting files with languages: %s, %s.", lang_code_1, lang_code_2)

    domain_groups = defaultdict(dict)
    for file in source_p.iterdir():
        if match := file_group_regex.match(file.stem):
            domain = match.group("domain")
            lang_code = match.group("lang_code")
            domain_groups[domain][lang_code] = file
        else:
            logger.debug(
                f"File {file.name} does not match expected pattern and will be skipped."
            )

    logger.debug("Filtering out languages not containing both langs")

    domain_groups_filtered = {
        domain: langs
        for domain, langs in domain_groups.items()
        if set([lang_code_1, lang_code_2]).issubset(langs.keys())
    }

    # TODO: Collected files: ...
    logger.info(f"Collected domains: {len(domain_groups_filtered)}")

    logger.info("Processing domains...")

    for domain, langs in domain_groups_filtered.items():
        logger.debug(f"Processing domain: {domain}")

        df_lang_code_1 = pd.read_json(langs[lang_code_1], lines=True)
        df_lang_code_1 = df_lang_code_1[
            ~df_lang_code_1["url"].apply(contains_dates_or_many_numbers)
        ]

        df_lang_code_2 = pd.read_json(langs[lang_code_2], lines=True)
        df_lang_code_2 = df_lang_code_2[
            ~df_lang_code_2["url"].apply(contains_dates_or_many_numbers)
        ]

        result_df = compare_urls(
            df_lang_code_1, df_lang_code_2, scorer=fuzz.ratio, score_cutoff=97
        )

        logger.debug(f"Matches found for {domain} before filtering: {len(result_df)}")

        if len(result_df) > 0:
            result_df_filtered = result_df[
                ~result_df[f"{lang_code_1}_url"].apply(ends_with_digit)
                & ~result_df[f"{lang_code_2}_url"].apply(ends_with_digit)
            ]
            if len(result_df_filtered) > 0:
                logger.info(
                    f"Saving {len(result_df_filtered)} valid matches for {domain}."
                )
                result_df_filtered.to_csv(output_dir / f"{domain}.csv", index=False)
                continue

        logger.debug(
            f"No valid matches found for {domain} after filtering. Skipping file."
        )
