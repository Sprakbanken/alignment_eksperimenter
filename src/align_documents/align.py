from pathlib import Path
import pandas as pd
import logging
from sentence_transformers import SentenceTransformer, util
import torch
from align_documents.utils.split import chunk_texts
from align_documents.types import AggregationStrategy
import regex as re
from functools import partial
from tqdm import tqdm

logger = logging.getLogger(__name__)


def create_document_embeddings(
    embedding_model: SentenceTransformer,
    documents: list[str],
    aggregation_strategy: AggregationStrategy,
    batch_size: int,
) -> list[torch.Tensor]:
    match aggregation_strategy:
        case "cut-off":
            embeddings = embedding_model.encode(documents, batch_size=batch_size)
        case "mean":
            chunked_docs = chunk_texts(
                documents,
                embedding_model.tokenizer,
                embedding_model.get_max_seq_length(),
            )
            embeddings = [
                torch.mean(
                    embedding_model.encode(
                        doc_chunks,
                        batch_size=batch_size,
                        convert_to_tensor=True,
                        show_progress_bar=False,
                    ),
                    axis=0,
                )
                for doc_chunks in tqdm(chunked_docs, desc="Embedding documents")
            ]
        case _:
            raise ValueError("Invalid aggregation strategy")

    return embeddings


def get_document_embeddings(
    embedding_model: SentenceTransformer,
    documents: list[str],
    embedding_directory: Path,
    filename_identifier: str,
    aggregation_strategy: AggregationStrategy,
    batch_size: int,
) -> list[torch.Tensor]:
    """Get existing or create document embeddings"""
    filename = embedding_directory / f"{filename_identifier}_{aggregation_strategy}.pt"

    if filename.exists():
        logger.debug("Loading embeddings from %s", filename)
        embeddings = torch.load(filename)
    else:
        logger.debug("Creating embeddings for %s", filename)
        embeddings = create_document_embeddings(
            embedding_model, documents, aggregation_strategy, batch_size
        )
        torch.save(embeddings, f=filename)

    return embeddings


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


def align(
    df: pd.DataFrame,
    website_name: str,
    embedding_dir: Path | None,
    embedding_model: SentenceTransformer,
    match_threshold: float,
    aggregation_strategy: AggregationStrategy,
    batch_size: int,
    languages: tuple[str, str],
    min_doc_len: int | None,
    number_to_letter_ratio: float,
) -> pd.DataFrame:
    """Align documents using sentence embeddings."""

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
    matches = [
        (i, e[0])
        for i, e in enumerate(search_result)
        if e[0]["score"] > match_threshold
    ]
    logger.debug("Number of matches: %s", len(matches))

    if matches:
        lang1_indices = [i for i, _ in matches]
        lang2_indices = [e["corpus_id"] for _, e in matches]

        lang1_df = lang1_df.loc[lang1_indices]
        lang1_df.index = range(len(lang1_df))

        lang2_df = lang2_df.loc[lang2_indices]
        lang2_df.index = range(len(lang2_df))

        df = lang1_df.merge(
            lang2_df, on=lang1_df.index, suffixes=("_" + lang1, "_" + lang2)
        )
        logger.debug("Number of aligned documents: %s", len(df))
        return df

    return pd.DataFrame()
