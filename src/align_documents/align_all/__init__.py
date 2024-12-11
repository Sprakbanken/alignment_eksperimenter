from argparse import ArgumentParser
from pathlib import Path
import os
import pandas as pd
import logging
from tqdm import tqdm
from align_documents.utils import setup_logging
from align_documents.align import align
from align_documents.types import AggregationStrategy
import tomllib


logger = logging.getLogger(__name__)


def get_file_info(data_dir: Path) -> pd.DataFrame:
    info = sorted([e.name[:-6].split("_") + [e.name] for e in data_dir.iterdir()])
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


def validate_config(config: dict) -> dict:
    if config["data_dir"]:
        config["data_dir"] = Path(config.data_dir)
        if not config["data_dir"].exists():
            raise ValueError("Data directory does not exist.")
    else:
        data_dir = os.environ.get("MALFRID", None)
        if data_dir is None:
            raise ValueError(
                "No data directory provided (no data_dir in config file and no MALFRID environment variable)."
            )
        config["data_dir"] = Path(data_dir)

    config_keys = [
        "output_dir",
        "embedding_model",
        "embedding_dir",
        "log_level",
        "batch_size",
        "aggregation_strategy",
        "match_threshold",
        "languages",
    ]
    for key in config_keys:
        if key not in config:
            raise ValueError(f"Missing key {key} in config file.")

    config_casts = {
        "embedding_dir": lambda x: Path(x),
        "output_dir": lambda x: Path(x),
        "match_threshold": lambda x: float(x),
        "aggregation_strategy": lambda x: AggregationStrategy(x),
        "batch_size": lambda x: int(x),
        "languages": lambda x: tuple(x),
    }

    for key, cast_function in config_casts.items():
        config[key] = cast_function(config[key])

    if len(config["languages"]) != 2:
        raise ValueError("languages must (only) contain two languages")

    return config


def main():
    parser = ArgumentParser()
    parser.add_argument(
        "-c",
        "--config_file",
        help="Path to the config file",
        type=Path,
        default=Path(__file__).parent / "alignment_config.toml",
    )
    parser.add_argument("-l", "--log_level", help="Log level", default="INFO")
    args = parser.parse_args()
    setup_logging("align_all", args.log_level)

    with open(args.config_file, "rb") as f:
        config = tomllib.load(f)

    logger.info(config)
    validate_config(config)

    df = get_file_info(config["data_dir"])
    df = filter_df(df, languages=config["languages"])

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
            embedding_dir=config["embedding_dir"],
            model_id=config["embedding_model"],
            match_threshold=config["match_threshold"],
            aggregation_strategy=config["aggregation_strategy"],
            batch_size=config["batch_size"],
            languages=config["languages"],
        )
        logger.debug(aligned_documents)
        dfs.append(aligned_documents)

    aligned_docs = pd.concat(dfs)
    aligned_docs.index = range(len(aligned_docs))

    config["output_dir"].mkdir(exist_ok=True, parents=True)
    outfile = config["output_dir"] / "aligned_docs.jsonl"
    i = 0
    while outfile.exists():
        i += 1
        outfile = config["output_dir"] / f"aligned_docs_{i}.jsonl"

    aligned_docs.to_json(outfile, lines=True, orient="records")
    logger.info("Aligned documents saved to %s", outfile)
