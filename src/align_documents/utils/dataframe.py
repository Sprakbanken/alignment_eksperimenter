from pathlib import Path
import pandas as pd
import logging

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


def jsonl_files_to_df(source_dir: Path, filenames: pd.Series) -> pd.DataFrame:
    dfs = []
    for e in filenames:
        e = source_dir / e
        dfs.append(pd.read_json(e, lines=True))
    df = pd.concat(dfs)
    df.index = range(len(df))
    df["fulltext_joined"] = df.fulltext.apply(lambda x: "\n".join(x))
    return df
