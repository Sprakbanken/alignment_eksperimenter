import json
from logging import getLogger
from pathlib import Path

from tqdm import tqdm
import pandas as pd
from transformers import AutoTokenizer
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
import argparse

from align_documents.utils.dataframe import (
    get_file_info,
    jsonl_files_to_df,
)
from align_documents.utils.config import get_config
from align_documents.utils.logging import setup_logging

logger = getLogger(__name__)

INFO_FILENAME_FULL_DATA = "stats_per_doc.csv"
INFO_FILENAME_OVERVIEW = "overview.json"

# Level 0 multi-index groups
METADATA_COLS = "Metadata"
DATA_COLS = "Data"
STAT_COLS = "Stats"  # Stats calculated by this command


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
    data_dir: Path, tokenizer: PreTrainedTokenizerBase | None = None
) -> pd.DataFrame:
    files_df = get_file_info(data_dir)

    # Return values:
    stats_per_doc = pd.DataFrame()

    for website, df_ in tqdm(files_df.groupby("website"), "Calculating stats"):
        logger.debug("Calculating stats for website %s", website)
        website_df = jsonl_files_to_df(data_dir, df_["file_name"])
        data_cols = ["fulltext", "fulltext_joined"]

        # Group the columns to know which ones to process later.
        website_df = pd.concat(
            {
                METADATA_COLS: website_df.drop(data_cols, axis="columns"),
                DATA_COLS: website_df[data_cols],
            },
            axis="columns",
        )

        if tokenizer:
            # NB: This is very slow. Most of the info command's time is spent here.
            tokens_list = tokenizer(
                website_df[DATA_COLS, "fulltext_joined"].tolist(),
                # Minor speed optimizations:
                return_token_type_ids=False,
                return_attention_mask=False,
            )["input_ids"]

            assert isinstance(tokens_list, list)
            assert len(tokens_list) == len(website_df)

            token_counts = [len(tokens) for tokens in tokens_list]

            website_df[STAT_COLS, "fulltext_tokens"] = token_counts

        website_df[STAT_COLS, "fulltext_lines"] = website_df[
            DATA_COLS, "fulltext"
        ].apply(len)
        website_df[STAT_COLS, "fulltext_words"] = website_df[
            DATA_COLS, "fulltext_joined"
        ].apply(lambda text: len(text.split()))
        website_df[STAT_COLS, "fulltext_characters"] = website_df[
            DATA_COLS, "fulltext_joined"
        ].apply(len)

        website_df.drop(DATA_COLS, axis="columns", inplace=True)

        stats_per_doc = pd.concat([stats_per_doc, website_df])

    return stats_per_doc


def _print_overview(data: dict) -> None:
    print(json.dumps(data, indent=4))


def get_overview(stats_per_doc: pd.DataFrame) -> dict:
    def _get_stat_col_stats(df: pd.DataFrame) -> dict:
        if STAT_COLS not in stats_per_doc.columns:
            return {}

        return df[STAT_COLS].describe().to_dict()

    # Aggregate stats per site per lang

    stats_per_site = {}

    for site, per_site in stats_per_doc.groupby((METADATA_COLS, "domain")):
        site_overview = {
            "n_docs": len(per_site),
            "stats": _get_stat_col_stats(per_site),
            "langs": {},
        }

        for lang, per_lang in per_site.groupby((METADATA_COLS, "lang")):
            site_overview["langs"][lang] = {
                "n_docs": len(per_lang),
                "stats": _get_stat_col_stats(per_lang),
            }

        stats_per_site[site] = site_overview

    return {
        "n_domains": stats_per_doc[METADATA_COLS, "domain"].nunique(),
        "n_docs": len(stats_per_doc),
        "stats": _get_stat_col_stats(stats_per_doc),
        "sites": stats_per_site,
    }


def get_tokenizer(embedding_model: str):
    try:
        tokenizer = AutoTokenizer.from_pretrained(embedding_model)
    except Exception:
        logger.warning(
            "Failed to get tokenizer based on config's 'embedding_model'. Skipping model-dependent stats."
        )
        tokenizer = None

    return tokenizer


def get_args():
    parser = argparse.ArgumentParser(
        "Calculate statistics about source data for alignment"
    )
    parser.add_argument(
        "-c",
        "--config_file",
        help="Path to the config file for alignment",
        type=Path,
        default=Path("alignment_config.toml"),
    )
    parser.add_argument(
        "-p",
        "--print-overview",
        action="store_true",
        help="Print dataset overview/aggregate stats to console",
    )

    parser.add_argument(
        "-P",
        "--print-full",
        action="store_true",
        help="Print full dataset stats to console",
    )

    parser.add_argument(
        "-t",
        "--use-tokenizer",
        action="store_true",
        help=(
            "Include tokenizer-dependent stats."
            f" Has no effect if {INFO_FILENAME_FULL_DATA} is not being written."
            "\nWarning: may greatly increase processing time"
        ),
    )
    parser.add_argument("--log_level", choices=["DEBUG", "INFO"], default="INFO")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help=f"Overwrite already-existing {INFO_FILENAME_FULL_DATA}.",
    )
    return parser.parse_args()


def main():
    args = get_args()
    setup_logging("info", log_level=args.log_level)
    logger.info(args)

    config = get_config(args.config_file)
    logger.info(config)
    data_dir = config["data_dir"]

    if not data_dir.exists():
        raise FileNotFoundError("Provided data directory does not exist")

    config["output_dir"].mkdir(exist_ok=True, parents=True)

    # Full data
    stats_per_doc_path: Path = config["output_dir"] / INFO_FILENAME_FULL_DATA

    if not args.overwrite and stats_per_doc_path.exists():
        logger.info(f"Using already existing file: {stats_per_doc_path}")
        stats_per_doc = pd.read_csv(stats_per_doc_path, header=[0, 1])
    else:
        tokenizer = (
            get_tokenizer(config["embedding_model"]) if args.use_tokenizer else None
        )
        stats_per_doc = get_stats_per_doc(data_dir, tokenizer=tokenizer)
        stats_per_doc.to_csv(stats_per_doc_path, index=False)
        logger.info(f"Full data saved to `{stats_per_doc_path}`")

    # Aggregate stats
    overview_path: Path = config["output_dir"] / INFO_FILENAME_OVERVIEW

    overview = get_overview(stats_per_doc)
    with open(overview_path, "w") as f:
        f.write(json.dumps(overview))
        logger.info(f"Overview saved to `{overview_path}`")

    if args.print_overview:
        _print_overview(overview)
