from pathlib import Path
import pandas as pd
import logging
from tqdm import tqdm
from align_documents.utils.get_embedding_model import get_embedding_model
from align_documents.align import align


logger = logging.getLogger(__name__)


def get_file_info(data_dir: Path) -> pd.DataFrame:
    info = sorted([e.name[:-6].split("_") + [e.name] for e in data_dir.glob("*.jsonl")])
    df = pd.DataFrame(info, columns=["website", "language", "format", "file_name"])
    logger.info("Found %s files in %s", len(df), data_dir)
    logger.info("Number of unique websites: %s", len(df.website.unique()))
    logger.info("Unique languages:          %s", df.language.unique())
    logger.info("Unique formats:            %s", df.format.unique())
    return df


def filter_df(df: pd.DataFrame, languages: tuple[str, str]) -> pd.DataFrame:
    """Return a DataFrame containing only rows with websites that has files in both the specified languages."""
    multilingual_websites = df.groupby("website").filter(
        lambda x: len(x.language.unique()) > 1
    )
    single_language_websites = df.groupby("website").filter(
        lambda x: len(x.language.unique()) == 1
    )
    logger.info(
        "Number of websites with multiple languages: %s",
        len(multilingual_websites.website.unique()),
    )
    logger.info(
        "Number of websites with only one language: %s",
        len(single_language_websites.website.unique()),
    )

    websites_to_remove = []
    for website, df_ in multilingual_websites.groupby("website"):
        if not set(languages) - set(df_.language.unique()) == set():
            logger.debug(
                "Website %s does not have files in both languages %s and %s",
                website,
                *languages,
            )
            websites_to_remove.append(website)

    multilingual_websites = multilingual_websites[
        ~multilingual_websites.website.isin(websites_to_remove)
    ]
    return multilingual_websites


def read_all_jsonl_files(source_dir: Path, filenames: pd.Series) -> pd.DataFrame:
    dfs = []
    for e in filenames:
        e = source_dir / e
        dfs.append(pd.read_json(e, lines=True))
    df = pd.concat(dfs)
    df.index = range(len(df))
    df["fulltext_joined"] = df.fulltext.apply(lambda x: "\n".join(x))
    return df


def main(args, config):
    df = get_file_info(config["data_dir"])
    df = filter_df(df, languages=config["languages"])

    embedding_model = get_embedding_model(config["embedding_model"])

    embedding_directory: Path = config["embedding_dir"] / config["embedding_model"]
    embedding_directory.mkdir(exist_ok=True, parents=True)

    dfs = []
    for website, df_ in tqdm(
        df.groupby("website"),
        total=len(df.website.unique()),
        desc="Processing files per website",
    ):
        logger.debug("Processing website %s", website)
        logger.debug("Number of files: %s", len(df_))
        logger.debug("Formats: %s", df_.format.unique())

        all_website_docs = read_all_jsonl_files(
            source_dir=config["data_dir"], filenames=df_.file_name
        )
        logger.debug("Number of documents: %s", len(all_website_docs))

        aligned_documents = align(
            all_website_docs,
            website_name=website,
            embedding_dir=embedding_directory,
            embedding_model=embedding_model,
            match_threshold=config["match_threshold"],
            aggregation_strategy=config["aggregation_strategy"],
            batch_size=config["batch_size"],
            languages=config["languages"],
            min_doc_len=config["min_document_length"],
            number_to_letter_ratio=config["number_to_letter_ratio"],
        )
        dfs.append(aligned_documents)

    aligned_docs = pd.concat(dfs)
    aligned_docs.index = range(len(aligned_docs))

    logger.info("Number of aligned documents: %s", len(aligned_docs))

    config["output_dir"].mkdir(exist_ok=True, parents=True)
    outfile = config["output_dir"] / "aligned_docs.jsonl"
    i = 0
    while outfile.exists():
        i += 1
        outfile = config["output_dir"] / f"aligned_docs_{i}.jsonl"

    aligned_docs.to_json(outfile, lines=True, orient="records")
    logger.info("Aligned documents saved to %s", outfile)
