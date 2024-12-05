from argparse import ArgumentParser
from pathlib import Path
import os
import pandas as pd
import logging
from tqdm import tqdm
from align_documents.utils import setup_logging
from align_documents.align import align


logger = logging.getLogger(__name__)


def get_file_info(data_dir: Path) -> pd.DataFrame:
    info = sorted([e.name[:-6].split("_") + [e.name]
                  for e in data_dir.iterdir()])
    df = pd.DataFrame(
        info, columns=["website", "language", "format", "file_name"])
    logger.info("Found %s files in %s", len(df), data_dir)
    logger.info("Number of unique websites: %s", len(df.website.unique()))
    logger.info("Unique languages:          %s", df.language.unique())
    logger.info("Unique formats:            %s", df.format.unique())
    return df


def filter_df(df: pd.DataFrame, languages: tuple[str, str]) -> pd.DataFrame:
    """Return a DataFrame containing only rows with websites that has files in both the specified languages."""
    multilingual_websites = df.groupby("website").filter(
        lambda x: len(x.language.unique()) > 1)
    single_language_websites = df.groupby("website").filter(
        lambda x: len(x.language.unique()) == 1)
    logger.info("Number of websites with multiple languages: %s",
                len(multilingual_websites.website.unique()))
    logger.info("Number of websites with only one language: %s",
                len(single_language_websites.website.unique()))

    websites_to_remove = []
    for website, df_ in multilingual_websites.groupby("website"):
        if not set(languages)-set(df_.language.unique()) == set():
            logger.debug(
                "Website %s does not have files in both languages %s and %s", website, *languages)
            websites_to_remove.append(website)

    multilingual_websites = multilingual_websites[~multilingual_websites.website.isin(
        websites_to_remove)]
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


def main():
    parser = ArgumentParser()
    parser.add_argument(
        "-d", "--data_dir", help="Path to the directory containing documents to align.", type=Path)
    parser.add_argument("-od", "--output_dir",
                        help="Path to the output directory where aligned docs will be stored.", type=Path, default=Path("output"))
    parser.add_argument("-em", "--embedding_model",
                        help="Sentence embedding model to use (local path or huggingface hub repo id).", default="NbAiLab/nb-sbert-base")
    parser.add_argument("-ed", "--embedding_dir",
                        help="Directory to store/read sentence embeddings.", type=Path, default=Path("embeddings"))
    parser.add_argument("-l", "--log_level", help="Set the log level.",
                        default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])
    parser.add_argument("-mt", "--match_threshold", help="Threshold for matching documents.",
                        type=float, default=0.95)
    parser.add_argument("-bs", "--batch_size",
                        help="Batch size for encoding sentences.", type=int, default=32)
    parser.add_argument("-as", "--aggregation_strategy",
                        help="Aggregation strategy for sentence embeddings.", choices=["cut-off", "max", "mean"], default="cut-off")
    args = parser.parse_args()

    setup_logging("align_all", args.log_level)

    if args.data_dir:
        if not args.data_dir.exists():
            raise ValueError("Data directory does not exist.")
    else:
        data_dir = os.environ.get("MALFRID", None)
        if not data_dir:
            raise ValueError(
                "No data directory provided (no --data_dir argument and no MALFRID environment variable).")
        args.data_dir = Path(data_dir)

    logger.info(args)

    df = get_file_info(args.data_dir)
    df = filter_df(df, languages=("nob", "nno"))

    dfs = []
    for website, df_ in tqdm(df.groupby("website"), total=len(df.website.unique()), desc="Processing files per website"):
        logger.debug("Processing website %s", website)
        logger.debug("Number of files: %s", len(df_))
        logger.debug("Formats: %s", df_.format.unique())

        all_website_docs = read_all_jsonl_files(
            source_dir=args.data_dir, filenames=df_.file_name)
        logger.debug("Number of documents: %s", len(all_website_docs))

        aligned_documents = align(
            all_website_docs, website_name=website, embedding_dir=args.embedding_dir, model_id=args.embedding_model, match_threshold=args.match_threshold, aggregation_strategy=args.aggregation_strategy, batch_size=args.batch_size)
        logger.debug(aligned_documents)
        dfs.append(aligned_documents)

    aligned_docs = pd.concat(dfs)
    aligned_docs.index = range(len(aligned_docs))

    args.output_dir.mkdir(exist_ok=True, parents=True)
    outfile = args.output_dir / "aligned_docs.jsonl"
    i = 0
    while outfile.exists():
        i += 1
        outfile = args.output_dir / f"aligned_docs_{i}.jsonl"

    aligned_docs.to_json(outfile, lines=True, orient="records")
    logger.info("Aligned documents saved to %s", outfile)
