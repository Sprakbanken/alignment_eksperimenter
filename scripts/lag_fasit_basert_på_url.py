import argparse
import logging
import re
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
from rapidfuzz import fuzz

from align_documents.utils.config import get_config
from align_documents.utils.logging import setup_logging

logger = logging.getLogger(__name__)

# Move all function definitions here, outside of if __name__ == "__main__"
def extract_language_code(url, lang_code_regex):
    match = lang_code_regex.search(url)
    return match.group(1) if match else None

def normalize_url(url, lang_code_regex):
    return re.sub(r'/[a-z]{2}-[A-Z]{2}/', '/xx-XX/', url) if extract_language_code(url, lang_code_regex) else url

def is_valid_url_pair(url1, url2, lang_code_regex, threshold=97):
    return fuzz.ratio(normalize_url(url1, lang_code_regex), normalize_url(url2, lang_code_regex)) >= threshold

def ends_with_digit(string, mimetype):
    match = re.search(rf'(\d+)(?=\.{mimetype}$)', string)
    return bool(match) if match else False

def match_rows(nn_row, df_nob_rows, lang_code_1, lang_code_2, lang_code_regex):
    matches = []
    nn_url = getattr(nn_row, 'url')
    for nb_row in df_nob_rows:
        nb_url = getattr(nb_row, 'url')
        if is_valid_url_pair(nn_url, nb_url, lang_code_regex):
            matches.append({
                f"{lang_code_1}_doc_hash": getattr(nn_row, 'doc_hash'),
                f"{lang_code_2}_doc_hash": getattr(nb_row, 'doc_hash'),
                f"{lang_code_1}_url": nn_url,
                f"{lang_code_2}_url": nb_url,
                f"{lang_code_1}_fulltext": getattr(nn_row, 'fulltext'),
                f"{lang_code_2}_fulltext": getattr(nb_row, 'fulltext')
            })
    return matches

def contains_dates_or_many_numbers(url):
    return (
        bool(re.search(r'/\d{4}(/|$)', url)) or  #match years
        bool(re.search(r'\d{6,}', url)) or      #match long sequences of digits
        bool(re.search(r'/\d+/?$', url)) or       #match a number at the end (e.g 06/)
        bool(re.search(r'/[^/]*\d+/?$', url))   #match just a digit at the end (e.g 06)
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create document pairs based on URL similarity')
    parser.add_argument('--log_level', default='INFO', 
                      choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                      help='Set the logging level')
    args = parser.parse_args()
    setup_logging("make_pairs_based_on_url", args.log_level)
    config = get_config(config_file="alignment_config.toml")

    source_p = config.get("data_dir")

    languages = config.get("languages")

    lang_code_1= languages[0]
    lang_code_2 = languages[1]

    mimetype = "html"

    output_dir = config.get("output_dir")
    output_dir = output_dir / "url_pairs"
    output_dir.mkdir(exist_ok=True, parents=True)

    
    file_group_regex = re.compile(rf"(.*?)(?:_{lang_code_1}|_{lang_code_2})_{mimetype}")
    lang_code_regex = re.compile(r'/([a-z]{2}-[A-Z]{2}|nynorsk)/')

    all_files = source_p.iterdir()
    domain_groups = defaultdict(list)
    for file in all_files:
        match = file_group_regex.match(file.stem)
        if match:
            domain_groups[match.group(1)].append(file)
        else:
            logger.warning(f"File {file.name} does not match expected pattern and will be skipped.")

    grouped_files = [
        (domain, tuple(files))
        for domain, files in domain_groups.items()
        if len(files) > 1
    ]
    logger.info(f"Grouped domains: {len(grouped_files)}")

    #process each domain
    for domain, files in grouped_files:
        logger.info(f"Processing domain: {domain}")
        df_lang_code_1 = pd.DataFrame()
        df_lang_code_2 = pd.DataFrame()

        for file in files:
            if lang_code_1 in file.name:
                df_lang_code_1 = pd.read_json(file, lines=True)
                df_lang_code_1 = df_lang_code_1[~df_lang_code_1['url'].apply(contains_dates_or_many_numbers)]
            elif lang_code_2 in file.name:
                df_lang_code_2 = pd.read_json(file, lines=True)
                df_lang_code_2 = df_lang_code_2[~df_lang_code_2['url'].apply(contains_dates_or_many_numbers)]

        df_lang_code_2_rows = list(df_lang_code_2.itertuples())
        all_matches = []

        with ThreadPoolExecutor() as executor:
            results = executor.map(lambda row: match_rows(row, df_lang_code_2_rows, lang_code_1, lang_code_2, lang_code_regex), df_lang_code_1.itertuples())
            for matched_rows in results:
                all_matches.extend(matched_rows)

        result_df = pd.DataFrame(all_matches)
        logger.info(f"Matches found for {domain}: {len(result_df)}")

        if len(result_df) > 0:
            result_df_filtered = result_df[
            ~result_df[f"{lang_code_1}_url"].apply(ends_with_digit, mimetype) &
            ~result_df[f"{lang_code_2}_url"].apply(ends_with_digit, mimetype)
    ]
        else:
            result_df_filtered = pd.DataFrame()
        if len(result_df_filtered) > 0:
            result_df_filtered.to_csv(output_dir /  f"{lang_code_1}_{lang_code_2}" / f"{domain}.csv", index=False)
        else:
            continue