from pathlib import Path
import pandas as pd
import logging
from functools import partial
from collections.abc import Callable, Iterable
import regex as re


logger = logging.getLogger(__name__)


def get_file_info(data_dir: Path) -> pd.DataFrame:
    info = sorted([e.name[:-6].split("_") + [e.name, e] for e in data_dir.glob("*.jsonl")])
    df = pd.DataFrame(info, columns=["website", "language", "format", "file_name", "file_path"])
    logger.info("Found %s files in %s", len(df), data_dir)
    logger.info("Number of unique websites: %s", len(df.website.unique()))
    logger.info("Unique languages:          %s", df.language.unique())
    logger.info("Unique formats:            %s", df.format.unique())
    return df


def get_websites_with_both_langs(
    df: pd.DataFrame, languages: tuple[str, str]
) -> pd.DataFrame:
    """Return a DataFrame containing only rows with websites that has files in both the specified languages."""
    lang1, lang2 = languages
    return df.groupby("website").filter(
        lambda df: lang1 in df.language.unique() and lang2 in df.language.unique()
    )


def jsonl_files_to_df(filepaths: Iterable[str | Path]) -> pd.DataFrame:
    dfs = []
    for path in filepaths:
        logger.debug("Reading into dataframe: %s", path)
        dfs.append(pd.read_json(e, lines=True))
    logger.debug("Finished reading jsonl files into dataframes")
    df = pd.concat(dfs).reset_index(drop=True)
    df["fulltext_joined"] = df.fulltext.apply(lambda x: "\n".join(x))
    return df


def has_bad_quality(
    doc_text: str, min_len: int | None, number_to_letter_ratio: float
) -> bool:
    if min_len and len(doc_text) < min_len:
        return True
    num_nums = len(re.findall(r"\d", doc_text))
    num_letters = len(re.findall(r"[A-Za-zÅåÆæØø]", doc_text))
    if num_letters == 0:
        return True
    if num_nums / num_letters > number_to_letter_ratio:
        return True
    return False


def deduplicate_and_filter_on_quality(
    df: pd.DataFrame,
    quality_function: Callable[[str], bool],
    text_col: str = "fulltext_joined",
) -> pd.DataFrame:
    logger.debug("Number of documents before filtering: %s", len(df))
    df = df.drop_duplicates(subset=text_col)
    logger.debug("Number of documents after dropping duplicates: %s", len(df))
    df = df[~df[text_col].apply(quality_function)].reset_index(drop=True)
    logger.debug("Number of documents after filtering on quality: %s", len(df))
    return df


def get_lang1_lang2_dataframes(
    df: pd.DataFrame,
    languages: tuple[str, str],
    min_doc_len: int | None,
    number_to_letter_ratio: float,
) -> tuple[str, pd.DataFrame, str, pd.DataFrame]:
    """Split dataframe into"""

    quality_function = partial(
        has_bad_quality,
        min_len=min_doc_len,
        number_to_letter_ratio=number_to_letter_ratio,
    )

    lang1, lang2 = languages

    lang1_df = df[df.lang == lang1]
    lang2_df = df[df.lang == lang2]

    lang1_df = deduplicate_and_filter_on_quality(
        lang1_df, quality_function=quality_function
    )
    lang2_df = deduplicate_and_filter_on_quality(
        lang2_df, quality_function=quality_function
    )

    if len(lang1_df) > len(lang2_df):
        # Set lang1 to be language with fewest documents (for semantic search below)
        lang1, lang2 = lang2, lang1
        lang1_df, lang2_df = lang2_df, lang1_df

    return (lang1, lang1_df, lang2, lang2_df)
