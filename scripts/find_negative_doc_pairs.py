from align_documents.utils.logging import setup_logging
from align_documents.utils.get_embedding_model import get_embedding_model
from align_documents.utils.config import get_config
from align_documents.utils.dataframe import (
    jsonl_files_to_df,
    get_websites_with_both_langs,
    get_file_info,
)
from align_documents.align import get_document_embeddings, has_bad_quality
from align_documents.types import AggregationStrategy

from pathlib import Path
import logging
from argparse import ArgumentParser
import pandas as pd
from functools import partial
from sentence_transformers import util

from tqdm import tqdm

logger = logging.getLogger(__name__)


def find_negative_doc_pairs(
    df: pd.DataFrame,
    min_threshold: float,
    max_threshold: float,
    website_name: str,
    embedding_dir: Path | None,
    aggregation_strategy: AggregationStrategy,
    batch_size: int,
    languages: tuple[str, str],
    min_doc_len: int | None,
    number_to_letter_ratio: float,
    pairs_per_website: int,
):
    lang1, lang2 = languages

    quality_function = partial(
        has_bad_quality,
        min_len=min_doc_len,
        number_to_letter_ratio=number_to_letter_ratio,
    )

    lang1_df = df[df.lang == lang1]
    logger.debug("Number of documents in %s before filtering: %s", lang1, len(lang1_df))
    lang1_df = lang1_df.drop_duplicates(subset="fulltext_joined")
    logger.debug(
        "Number of documents in %s after dropping duplicates: %s", lang1, len(lang1_df)
    )
    lang1_df = lang1_df[~lang1_df.fulltext_joined.apply(quality_function)]
    logger.debug(
        "Number of documents in %s after filtering on quality: %s", lang1, len(lang1_df)
    )

    lang1_df.index = range(len(lang1_df))

    lang2_df = df[df.lang == lang2]
    logger.debug("Number of documents in %s before filtering: %s", lang2, len(lang2_df))
    lang2_df = lang2_df.drop_duplicates(subset="fulltext_joined")
    logger.debug(
        "Number of documents in %s after dropping duplicates: %s", lang2, len(lang2_df)
    )
    lang2_df = lang2_df[~lang2_df.fulltext_joined.apply(quality_function)]
    logger.debug(
        "Number of documents in %s after filtering on quality: %s", lang2, len(lang2_df)
    )
    lang2_df.index = range(len(lang2_df))

    if len(lang1_df) > len(lang2_df):
        # Set lang1 to be language with fewest documents (for semantic search below)
        lang1, lang2 = lang2, lang1
        lang1_df, lang2_df = lang2_df, lang1_df

    lang1_embeddings = get_document_embeddings(
        embedding_model=embedding_model,
        documents=lang1_df.fulltext_joined,
        embedding_directory=embedding_dir,
        filename_identifier=f"{website_name}_{lang1}",
        aggregation_strategy=aggregation_strategy,
        batch_size=batch_size,
    )

    lang2_embeddings = get_document_embeddings(
        embedding_model=embedding_model,
        documents=lang2_df.fulltext_joined,
        embedding_directory=embedding_dir,
        filename_identifier=f"{website_name}_{lang2}",
        aggregation_strategy=aggregation_strategy,
        batch_size=batch_size,
    )

    search_result = util.semantic_search(lang1_embeddings, lang2_embeddings, top_k=1)
    negative_pairs = []
    for i, e in enumerate(search_result):
        if e[0]["score"] < min_threshold:
            continue
        if e[0]["score"] > max_threshold:
            continue
        negative_pairs.append((i, e[0]))
        if len(negative_pairs) >= pairs_per_website:
            break
    if negative_pairs:
        lang1_indices = [i for i, _ in negative_pairs]
        lang2_indices = [e["corpus_id"] for _, e in negative_pairs]

        lang1_df = lang1_df.loc[lang1_indices]
        lang1_df.index = range(len(lang1_df))

        lang2_df = lang2_df.loc[lang2_indices]
        lang2_df.index = range(len(lang2_df))

        df = lang1_df.merge(
            lang2_df, on=lang1_df.index, suffixes=("_" + lang1, "_" + lang2)
        )
        logger.debug("Number of negative pairs: %s", len(df))
        return df
    return pd.DataFrame()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "-c",
        "--config_file",
        help="Path to the config file",
        type=Path,
        default=Path("alignment_config.toml"),
    )
    parser.add_argument("-l", "--log_level", help="Log level", default="INFO")
    parser.add_argument(
        "--min_threshold",
        help="Minimum similarity threshold for negative pairs",
        type=float,
        default=0.5,
    )
    parser.add_argument(
        "--pairs_per_website",
        help="Number of negative pairs to generate per website",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--total_pairs",
        help="Total number of negative pairs to generate",
        type=int,
        default=100,
    )
    args = parser.parse_args()
    setup_logging("find_negative_doc_pairs", args.log_level)

    config = get_config(args.config_file)
    logger.info(
        "Negative pairs config: min_threshold=%s, pairs_per_website=%s, total_pairs=%s",
        args.min_threshold,
        args.pairs_per_website,
        args.total_pairs,
    )

    df = get_file_info(config["data_dir"])
    df = get_websites_with_both_langs(df, languages=config["languages"])

    embedding_model = get_embedding_model(config["embedding_model"])

    embedding_directory: Path = config["embedding_dir"] / config["embedding_model"]
    embedding_directory.mkdir(exist_ok=True, parents=True)

    dfs = []
    tot_len = 0
    for website, df_ in tqdm(
        df.groupby("website"),
        total=len(df.website.unique()),
        desc="Processing files per website",
    ):
        logger.debug("Processing website %s", website)
        logger.debug("Number of files: %s", len(df_))
        logger.debug("Formats: %s", df_.format.unique())

        all_website_docs = jsonl_files_to_df(
            source_dir=config["data_dir"], filenames=df_.file_name
        )
        logger.debug("Number of documents: %s", len(all_website_docs))

        negative_pairs_df = find_negative_doc_pairs(
            df=all_website_docs,
            min_threshold=args.min_threshold,
            max_threshold=config["match_threshold"],
            website_name=website,
            embedding_dir=embedding_directory,
            aggregation_strategy=config["aggregation_strategy"],
            batch_size=config["batch_size"],
            languages=config["languages"],
            min_doc_len=config["min_document_length"],
            number_to_letter_ratio=config["number_to_letter_ratio"],
            pairs_per_website=args.pairs_per_website,
        )
        tot_len += len(negative_pairs_df)
        dfs.append(negative_pairs_df)
        if tot_len > args.total_pairs:
            break
    negative_pairs_df = pd.concat(dfs, ignore_index=True)
    logger.info("Number of negative pairs: %s", len(negative_pairs_df))
    config["output_dir"].mkdir(exist_ok=True, parents=True)
    outfile = config["output_dir"] / "negative_pairs.jsonl"
    i = 0
    while outfile.exists():
        i += 1
        outfile = config["output_dir"] / f"negative_pairs_{i}.jsonl"

    negative_pairs_df.to_json(outfile, lines=True, orient="records", index=False)
    logger.info("Negative pairs saved to %s", outfile)
