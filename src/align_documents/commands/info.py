from __future__ import annotations

import json
from logging import getLogger
from typing import Iterable
from pathlib import Path

from tqdm import tqdm
import pandas as pd
from transformers import AutoTokenizer
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from align_documents.commands.align_all import (
    read_all_jsonl_files,
    get_file_info,
)


logger = getLogger(__name__)


# .jsonl example line (input-dataset):
#
# {
#   "doc_hash": "c2c2f4ddb713987517a1e16a80282e94cb68a970",
#   "lang": "nno",
#   "url": "https://www.yrkesfisker.no/aktuelt/nyhetsarkiv/lengdereduksjon-fartoy-over-under-8-meter/",
#   "domain": "yrkesfisker.no",
#   "date": "2024-01-05T14:30:30Z",
#   "mimetype": "html",
#   "fulltext": [
#     "Fiskefartøy som reduserer lengda for å komme under 8 meter",
#     "Den siste tida har Sjøfartsdirektoratet fått fleire spørsmål ...",
#   ]
# }


def get_stats_per_doc(
    data_dir: Path,
    tokenizer: PreTrainedTokenizerBase | None = None
) -> tuple[pd.DataFrame, Iterable]:

    files_df = get_file_info(data_dir)

    # Return values:
    stats_per_doc = pd.DataFrame()
    data_columns = set()

    for website, df_ in tqdm(files_df.groupby("website"), "Calculating stats"):
        website_df = read_all_jsonl_files(data_dir, df_["file_name"])
        initial_columns = website_df.columns

        if tokenizer:
            # NB: This is very slow. Most of the info command's time is spent here.
            tokens_list = tokenizer(
                website_df["fulltext_joined"].tolist(),
                # Minor speed optimizations:
                return_token_type_ids=False,
                return_attention_mask=False,
            )['input_ids']

            assert isinstance(tokens_list, list)
            assert len(tokens_list) == len(website_df)

            token_counts = [len(tokens) for tokens in tokens_list]

            website_df["fulltext_tokens"] = pd.Series(index=website_df.index)
            website_df["fulltext_tokens"] = token_counts

        website_df["fulltext_lines"] = pd.Series([len(lines) for lines in website_df["fulltext"]])
        website_df["fulltext_words"] = pd.Series(len(text.split()) for text in website_df["fulltext_joined"])
        website_df["fulltext_characters"] = pd.Series(len(text) for text in website_df["fulltext_joined"]) # TODO: Find out if outputted `\n`s are encoded or not

        website_df.drop(["fulltext", "fulltext_joined"], axis='columns', inplace=True)

        stats_per_doc = pd.concat([stats_per_doc, website_df])
        # We do this for every website, rather than once after the loop,
        # in case some websites differ in initial columns.
        data_columns.update(website_df.columns.difference(initial_columns))

    return stats_per_doc, list(data_columns)

def print_data(data: dict | pd.DataFrame) -> None:
    if isinstance(data, dict):
        print(json.dumps(data, indent=4))
    elif isinstance(data, pd.DataFrame):
        json_str = data.to_json(orient='records')
        assert json_str is not None

        # Simple workaround because DataFrame.to_dict() doesn't convert
        # `Timestamp`s to json-serializable values
        print(json.dumps(json.loads(json_str), indent=4))

def get_overview(
    stats_per_doc: pd.DataFrame,
    data_columns: Iterable | None = None
) -> dict:
    def _get_data_col_stats(df: pd.DataFrame, cols: Iterable | None) -> dict:
        if cols is None:
            return {}

        return df[cols].describe().to_dict()

    # Aggregate stats per site per lang

    sites = stats_per_doc["domain"].value_counts()
    stats_per_site = {}

    for site, per_site in stats_per_doc.groupby("domain"):
        site_overview = {
            "n_docs": len(per_site),
            "stats": _get_data_col_stats(per_site, data_columns),
            "langs": {},
        }

        for lang, per_lang in per_site.groupby("lang"):
            site_overview["langs"][lang] = {
                "n_docs": len(per_lang),
                "stats": _get_data_col_stats(per_lang, data_columns)
            }

        stats_per_site[site] = site_overview

    return {
        "n_sites": len(sites),
        "n_docs": len(stats_per_doc),
        "stats": _get_data_col_stats(stats_per_doc, data_columns),
        "sites": stats_per_site,
    }

def main(args, config):
    # TODO: Move config and arg verification into shared entry-point or get_config (in other modules too)
    #           --data-dir arg currently doesn't get caught by get_config verification

    data_dir = Path(args.data_dir or config["data_dir"])

    if not data_dir.exists():
        raise FileNotFoundError("Provided data directory does not exist")

    config["output_dir"].mkdir(exist_ok=True, parents=True)

    try:
        tokenizer = AutoTokenizer.from_pretrained(config["embedding_model"])
    except:
        logger.warning("Failed to get tokenizer based on config's 'embedding_model'. Skipping model-dependent stats.")
        tokenizer = None

    # Full data
    stats_per_doc, data_columns = get_stats_per_doc(data_dir, tokenizer=tokenizer)
    stats_per_doc_path = config["output_dir"]/"stats_per_doc.jsonl"

    stats_per_doc.to_json(stats_per_doc_path, lines=True, orient="records")
    logger.info(f"Full data saved to `{stats_per_doc_path}`")

    if args.print_full:
        print_data(stats_per_doc)

    # Aggregates
    overview = get_overview(stats_per_doc, data_columns=data_columns)
    overview_path = config["output_dir"]/"overview.json"

    with open(overview_path, "w") as f:
        f.write(json.dumps(overview))
        logger.info(f"Overview saved to `{overview_path}`")

    if args.print_overview:
        print_data(overview)
        print(f"\n- Saved to `{overview_path}`") # Printing this here too for visibility

