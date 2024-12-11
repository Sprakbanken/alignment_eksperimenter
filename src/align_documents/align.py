from pathlib import Path
import pandas as pd
import logging
from sentence_transformers import SentenceTransformer, util
import torch
import numpy as np
from typing import Literal
from align_documents.utils.split import tokenize_and_split_text
from align_documents.types import AggregationStrategy

logger = logging.getLogger(__name__)


def create_sentence_embeddings(
    embedding_model_id: str,
    sentences: list[str],
    aggregation_strategy: AggregationStrategy,
    batch_size: int,
) -> np.array:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(embedding_model_id, device=device)

    basemodel_max_len = model[0].auto_model.config.max_position_embeddings
    if basemodel_max_len != model.get_max_seq_length():
        logger.info(
            "Setting max_seq_length to %s (was %s)",
            basemodel_max_len,
            model.get_max_seq_length(),
        )
        model.max_seq_length = basemodel_max_len

    match aggregation_strategy:
        case "cut-off":
            embeddings = model.encode(sentences, batch_size=batch_size)
        case "max":
            maxlen_parts = [
                tokenize_and_split_text(
                    text,
                    tokenizer=model.tokenizer,
                    model_max_len=model.get_max_seq_length(),
                )
                for text in sentences
            ]
            embeddings = [
                np.max(model.encode(text_parts, batch_size=batch_size), axis=0)
                for text_parts in maxlen_parts
            ]
        case "mean":
            maxlen_parts = [
                tokenize_and_split_text(
                    text,
                    tokenizer=model.tokenizer,
                    model_max_len=model.get_max_seq_length(),
                )
                for text in sentences
            ]
            embeddings = [
                np.mean(model.encode(text_parts, batch_size=batch_size), axis=0)
                for text_parts in maxlen_parts
            ]
        case _:
            raise ValueError("Invalid aggregation strategy")
    return embeddings


def get_sentence_embeddings(
    embedding_model_id: str,
    sentences: list[str],
    embedding_directory: Path,
    filename_identifier: str,
    aggregation_strategy: AggregationStrategy,
    batch_size: int,
) -> np.array:
    """Get existing or create sentence embeddings"""
    filename = embedding_directory / f"{filename_identifier}_{aggregation_strategy}.npy"

    if filename.exists():
        embeddings = np.load(filename)
    else:
        embeddings = create_sentence_embeddings(
            embedding_model_id, sentences, aggregation_strategy, batch_size
        )
        np.save(filename, embeddings)

    return embeddings


def align(
    df: pd.DataFrame,
    website_name: str,
    embedding_dir: Path | None,
    model_id: str,
    match_threshold: float,
    aggregation_strategy: AggregationStrategy,
    batch_size: int,
) -> pd.DataFrame:
    """Align documents using sentence embeddings."""
    nynorsk_df = df[df.lang == "nno"]
    nynorsk_df.index = range(len(nynorsk_df))

    bokmål_df = df[df.lang == "nob"]
    bokmål_df.index = range(len(bokmål_df))

    # TODO: filter out texts of bad quality
    # TODO: filter out duplicate documents

    logger.debug("Number of documents in nynorsk: %s", len(nynorsk_df))
    logger.debug("Number of documents in bokmål: %s", len(bokmål_df))

    embedding_directory = embedding_dir / model_id
    embedding_directory.mkdir(exist_ok=True, parents=True)

    nynorsk_embeddings = get_sentence_embeddings(
        embedding_model_id=model_id,
        sentences=nynorsk_df.fulltext_joined,
        embedding_directory=embedding_directory,
        filename_identifier=f"{website_name}_nynorsk",
        aggregation_strategy=aggregation_strategy,
        batch_size=batch_size,
    )

    bokmål_embeddings = get_sentence_embeddings(
        embedding_model_id=model_id,
        sentences=bokmål_df.fulltext_joined,
        embedding_directory=embedding_directory,
        filename_identifier=f"{website_name}_bokmål",
        aggregation_strategy=aggregation_strategy,
        batch_size=batch_size,
    )

    search_result = util.semantic_search(nynorsk_embeddings, bokmål_embeddings, top_k=1)
    matches = [
        (i, e[0])
        for i, e in enumerate(search_result)
        if e[0]["score"] > match_threshold
    ]
    logger.debug("Number of matches: %s", len(matches))

    if matches:
        nynorsk_indices = [i for i, _ in matches]
        bokmål_indices = [e["corpus_id"] for _, e in matches]

        nynorsk_df = nynorsk_df.loc[nynorsk_indices]
        nynorsk_df.index = range(len(nynorsk_df))

        bokmål_df = bokmål_df.loc[bokmål_indices]
        bokmål_df.index = range(len(bokmål_df))

        df = nynorsk_df.merge(
            bokmål_df, on=nynorsk_df.index, suffixes=("_nynorsk", "_bokmål")
        )
        logger.debug("Number of aligned documents: %s", len(df))
        return df

    return pd.DataFrame()
