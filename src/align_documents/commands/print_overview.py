from __future__ import annotations

import json
from logging import getLogger
from typing import Iterable
from pathlib import Path

from torch import Tensor
from tqdm import tqdm
import pandas as pd
from sentence_transformers import SentenceTransformer

from align_documents.commands.align_all import (
    get_embedding_model,
    read_all_jsonl_files,
    get_file_info,
)


logger = getLogger(__name__)


# TODO:
# - Make separate input overview and output/embedding/alignment overview
# - Embedding-output stats?

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

def _get_info_from_filenames(data_dir):
    # TODO: Probably remove this block.
    logger.info(f"Data directory: {data_dir}")
    documents_paths = [file for file in data_dir.iterdir() if file.suffix == ".jsonl"]
    if not documents_paths:
        raise FileNotFoundError(f"No .jsonl files found")

    # columns: ["website", "language", "format", "file_name"]
    return get_file_info(data_dir)

def get_stats_per_doc(
    data_dir: Path,
    embedding_model: SentenceTransformer | None = None
) -> tuple[pd.DataFrame, Iterable]:

    # TODO(1):
    # if embedding_model:
    #     embedding_model.max_seq_length = 0

    files_df = _get_info_from_filenames(data_dir)
    stats_per_doc = pd.DataFrame()

    for website, df_ in tqdm(files_df.groupby("website"), "Calculating stats"):
        website_df = read_all_jsonl_files(data_dir, df_["file_name"])

        if embedding_model:
            # TODO IMPORTANT: Implement this.
            pass
            # keys: ["input_ids", "token_type_ids", "attention_mask"]
            # TODO(1): - Need to tokenize the entire text here, in chunks.
            #              - By default doesn't support chunking; only cutoff?
            # tokens: dict[str, Tensor] = embedding_model.tokenize(website_df["fulltext_joined"])
            # token_lengths = [tensor.numel() for tensor in tokens["input_ids"]]
            # print(f"{token_lengths=}")
            # continue
            # # print(tokens)
            # token_counts = len(tokens["input_ids"])
            # website_df["n_tokens"] = pd.Series(index=website_df.index)
            # website_df["n_tokens"] = token_counts

        # TODO: Optimize this this if too slow
        website_df["fulltext_lines"] = pd.Series([len(lines) for lines in website_df["fulltext"]])
        website_df["fulltext_words"] = pd.Series(len(text.split()) for text in website_df["fulltext_joined"])
        website_df["fulltext_characters"] = pd.Series(len(text) for text in website_df["fulltext_joined"]) # TODO: Find out if outputted `\n`s are encoded or not

        website_df.drop(["fulltext", "fulltext_joined"], axis='columns', inplace=True)

        stats_per_doc = pd.concat([stats_per_doc, website_df])



    # TODO:  Maybe don't assume all files have the same data columns?
    sample_df = read_all_jsonl_files(data_dir, files_df["file_name"].iloc[0:1])
    data_columns = stats_per_doc.columns.difference(sample_df.columns)

    return stats_per_doc, data_columns

def print_overview(overview: dict) -> None:
    # TODO:
    #  - Better formatting:
    #       - indents
    #       - decimal points

    print(json.dumps(overview, indent=4))

def get_overview(
    stats_per_doc: pd.DataFrame,
    data_columns: Iterable | None = None
) -> dict:
    def _get_data_col_stats(df: pd.DataFrame, cols: Iterable | None) -> dict:
        if cols is None:
            return {}

        return df[cols].describe().to_dict()

    # Aggregate stats per site per lang
    #   TODO: - Optimize if slow

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

# @line_profiler.profile
def main(args, config):
    # TODO: --exclude & --include-only flags
    #           Example: --exclude language:eng
    #               Maybe json better due to url characters
    data_dir = Path(args.data_dir or config["data_dir"])

    # TODO 4 (?)
    if not data_dir.exists():
        raise FileNotFoundError("Provided data directory does not exist")

    # TODO 4: Move into shared entry-point (in other modules too)
    config["output_dir"].mkdir(exist_ok=True, parents=True)

    try:
        # TODO: Only get the tokenizer directly to save memory.
        embedding_model = get_embedding_model(config["embedding_model"])
    except:
        logger.warning("Failed to get embedding model from config file. Skipping model-dependent stats.")
        embedding_model = None

    # TODO 3 IMPORTANT: Add hash to filenames based on dataset hash
    #                   - Do we make dataset hashes already?

    # Full data
    stats_per_doc, data_columns = get_stats_per_doc(data_dir, embedding_model=embedding_model)
    stats_per_doc_path = config["output_dir"]/"stats_per_doc.jsonl"

    stats_per_doc.to_json(stats_per_doc_path, lines=True, orient="records") # TODO 3
    logger.info(f"Full data saved to `{stats_per_doc_path}`")

    # Aggregates
    overview = get_overview(stats_per_doc, data_columns=data_columns)
    overview_path = config["output_dir"]/"overview.json"

    with open(overview_path, "w") as f: # TODO 3
        f.write(json.dumps(overview))
        logger.info(f"Overview saved to `{overview_path}`")

    print_overview(overview)

    # Printing this here too for visibility
    print(f"\n- Saved to `{overview_path}`")
