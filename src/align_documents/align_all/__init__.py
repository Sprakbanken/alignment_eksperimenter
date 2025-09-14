from pathlib import Path
import logging
from tqdm import tqdm
from align_documents.utils import setup_logging, get_config, get_embedding_model
from align_documents.utils.config import CONFIG_PATH
from align_documents.utils.dataframe import (
    get_file_info,
    get_websites_with_both_langs,
    jsonl_files_to_df,
)
from align_documents.align import filter_and_align
import argparse

logger = logging.getLogger(__name__)


def get_args():
    parser = argparse.ArgumentParser(
        prog="align_documents", description=("Create and align document embeddings.")
    )

    parser.add_argument(
        "-c",
        "--config_file",
        help="Path to the config file for alignment",
        type=Path,
        default=Path("alignment_config.toml"),
    )
    parser.add_argument(
        "-l",
        "--log_level",
        help="Log level",
        default="INFO",
        choices=["INFO", "DEBUG", "WARNING", "ERROR"],
    )
    return parser.parse_args()


def main():
    args = get_args()
    setup_logging("align_all", log_level=args.log_level)
    logger.info(args)

    config = get_config(args.config_file)
    logger.info(config)

    df = get_file_info(config.data_dir)
    df = get_websites_with_both_langs(df, languages=config.languages)
    logger.info(
        "Number of websites with documents in both languages: %s", df.website.nunique()
    )
    logger.debug("Number of documents in both languages: %s", len(df))
    logger.debug(df.head(5))

    embedding_model = get_embedding_model(config.embedding_model)

    embedding_directory: Path = config.embedding_dir / config.embedding_model
    embedding_directory.mkdir(exist_ok=True, parents=True)

    output_dir = config.output_dir / "aligned"
    output_dir.mkdir(parents=True)

    # Save alignment config to output directory
    config_outfile = config.output_dir / "alignment_config.toml"
    config_outfile.write_text(args.config_file.read_text())

    lang_1, lang_2 = config.languages

    for website, df_ in tqdm(
        df.groupby("website"),
        total=len(df.website.unique()),
        desc="Processing files per website",
    ):
        logger.debug("Processing website %s", website)
        logger.debug("Number of files: %s", len(df_))
        logger.debug("Formats: %s", df_.format.unique())

        all_website_docs = jsonl_files_to_df(df_.file_path)
        logger.debug("Number of documents: %s", len(all_website_docs))

        aligned_documents = filter_and_align(
            all_website_docs,
            website_name=website,
            embedding_dir=embedding_directory,
            embedding_model=embedding_model,
            match_threshold=config.match_threshold,
            aggregation_strategy=config.aggregation_strategy,
            batch_size=config.batch_size,
            languages=config.languages,
            min_doc_len=config.min_document_length,
            number_to_letter_ratio=config.number_to_letter_ratio,
        )
        outfile = output_dir / f"{website}_{lang_1}_{lang_2}.jsonl"
        if not aligned_documents.empty:
            aligned_documents.to_json(
                outfile, lines=True, orient="records", index=False
            )

    logger.info("All aligned documents saved to %s", output_dir)
