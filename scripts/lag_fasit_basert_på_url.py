from pathlib import Path
import pandas as pd
import os
import re
from collections import defaultdict
from rapidfuzz import fuzz
from concurrent.futures import ThreadPoolExecutor
from align_documents.utils.config import get_config


if __name__ == "__main__":
    config = get_config(config_file="alignment_config.toml")

    source_p = config.get("data_dir")
    print("Source path", source_p)

    languages = config.get("languages")
    print("Languages", languages)

    lang_code_1= languages[0]
    lang_code_2 = languages[1]

    #file_group_regex = re.compile(r"(.*?)(?:_{lang_code1}|_{lang_code_2})_html")
    file_group_regex = re.compile(rf"(.*?)(?:_{lang_code_1}|_{lang_code_2})_html")
    lang_code_regex = re.compile(r'/([a-z]{2}-[A-Z]{2}|nynorsk)/')

    filer = [e for e in source_p.iterdir() if "html" in e.name and (lang_code_1 in e.name or lang_code_2 in e.name)]

    domain_groups = defaultdict(list)
    for file in filer:
        match = file_group_regex.match(file.stem)
        if match:
            domain_groups[match.group(1)].append(file)
        else:
            print(f"Regex did not match for file: {file}")

    grouped_files = [
        (domain, tuple(files))
        for domain, files in domain_groups.items()
        if len(files) > 1
    ]
    print(f"Grouped domains: {len(grouped_files)}")

    def extract_language_code(url):
        match = lang_code_regex.search(url)
        return match.group(1) if match else None

    def normalize_url(url):
        return re.sub(r'/[a-z]{2}-[A-Z]{2}/', '/xx-XX/', url) if extract_language_code(url) else url

    def is_valid_url_pair(url1, url2, threshold=97):
        return fuzz.ratio(normalize_url(url1), normalize_url(url2)) >= threshold

    def ends_with_digit(string):
        match = re.search(r'(\d+)(?=\.html$)', string)
        return bool(match) if match else False

    def match_rows(nn_row, df_nob_rows):
        matches = []
        nn_url = getattr(nn_row, 'url')
        for nb_row in df_nob_rows:
            nb_url = getattr(nb_row, 'url')
            if is_valid_url_pair(nn_url, nb_url):
                matches.append({
                    f"{lang_code_1}_doc_hash": getattr(nn_row, 'doc_hash'),
                    f"{lang_code_2}_doc_hash": getattr(nb_row, 'doc_hash'),
                    f"{lang_code_1}_url": nn_url,
                    f"{lang_code_2}_url": nb_url,
                    f"{lang_code_1}_fulltext": getattr(nn_row, 'fulltext'),
                    f"{lang_code_2}_fulltext": getattr(nb_row, 'fulltext')
                })
        return matches

    output_dir = Path("o")
    output_dir.mkdir(exist_ok=True)

    def contains_dates_or_many_numbers(url):
        return (
        bool(re.search(r'/\d{4}(/|$)', url)) or  #match years
        bool(re.search(r'\d{6,}', url)) or      #match long sequences of digits
        bool(re.search(r'/\d+/?$', url)) or       #match a number at the end (e.g 06/)
        bool(re.search(r'/[^/]*\d+/?$', url))   #match just a digit at the end (e.g 06)
    )
    #process each domain
    for domain, files in grouped_files:
        print(f"Processing: {domain}")
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
            results = executor.map(lambda row: match_rows(row, df_lang_code_2_rows), df_lang_code_1.itertuples())
            for matched_rows in results:
                all_matches.extend(matched_rows)

        result_df = pd.DataFrame(all_matches)
        print(f"Matches found for {domain}: {len(result_df)}")

        if len(result_df) > 0:
            result_df_filtered = result_df[
            ~result_df[f"{lang_code_1}_url"].apply(ends_with_digit) &
            ~result_df[f"{lang_code_2}_url"].apply(ends_with_digit)
    ]
        else:
            result_df_filtered = pd.DataFrame()
        if len(result_df_filtered) > 0:
            result_df_filtered.to_csv(output_dir / f"{domain}.csv", index=False)
        else:
            print(f"No valid matches found for {domain}.")
    
