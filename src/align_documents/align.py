from pathlib import Path
import pandas as pd
import logging
from sentence_transformers import SentenceTransformer, util
import torch
from align_documents.utils.split import chunk_texts
from align_documents.utils.dataframe import get_lang1_lang2_dataframes
from align_documents.types import AggregationStrategy
from collections.abc import Iterable

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
        embeddings = torch.load(filename, weights_only=False)

        # Crude check for compatible file version.
        #    TODO:
        #      Use hashes from hashtree for embedding filenames,
        #      then verify against hashes
        #          domain->lang->doc_hash: use lang hash
        #          lang->domain->doc_hash: use domain hash
        #          # TODO: Make hashes independent of tree structure?
        if len(embeddings) == len(documents):
            return embeddings
        else:
            logger.debug(
                "Number of existing embeddings differ from number of documents for %s. Regenerating...",
                filename_identifier,
            )

    logger.debug("Creating embeddings for %s", filename)
    embeddings = create_document_embeddings(
        embedding_model, documents, aggregation_strategy, batch_size
    )
    torch.save(embeddings, f=filename)

    return embeddings


def align(
    lang1_documents: Iterable[str],
    lang2_documents: Iterable[str],
    lang1_filename_identifier: str,
    lang2_filename_identifier: str,
    embedding_dir: Path,
    embedding_model: SentenceTransformer,
    match_threshold: float,
    aggregation_strategy: AggregationStrategy,
    batch_size: int,
) -> list[tuple[int, int]]:
    """Pair lang1_documents with lang2_documents by creating document embeddings
    with the given sentence transformers model and aggregation strategy.
    For each lang1 embedding, keep the most similar lang2 document over match_threshold, if any.
    Returns a list of tuples of (lang1_document_index, lang2_document_index)
    """
    lang1_embeddings = get_document_embeddings(
        embedding_model=embedding_model,
        documents=lang1_documents,
        embedding_directory=embedding_dir,
        filename_identifier=lang1_filename_identifier,
        aggregation_strategy=aggregation_strategy,
        batch_size=batch_size,
    )

    lang2_embeddings = get_document_embeddings(
        embedding_model=embedding_model,
        documents=lang2_documents,
        embedding_directory=embedding_dir,
        filename_identifier=lang2_filename_identifier,
        aggregation_strategy=aggregation_strategy,
        batch_size=batch_size,
    )

    search_result = util.semantic_search(lang1_embeddings, lang2_embeddings, top_k=1)
    matches = [
        (i, e[0]["corpus_id"])
        for i, e in enumerate(search_result)
        if e[0]["score"] > match_threshold
    ]
    logger.debug("Number of matches: %s", len(matches))
    return matches


def filter_and_align(
    df: pd.DataFrame,
    website_name: str,
    embedding_dir: Path,
    embedding_model: SentenceTransformer,
    match_threshold: float,
    aggregation_strategy: AggregationStrategy,
    batch_size: int,
    languages: tuple[str, str],
    min_doc_len: int,
    number_to_letter_ratio: float,
) -> pd.DataFrame:
    """Align documents using sentence embeddings.
    The dataframe 'df' contains documents in both languages, so a dataframe for each language is extracted from this.
    Then, the documents are embedded with a sentence embedding model.
    Documents in both languages that are similar above a certain threshold are kept
    and returned as a dataframe where each row contains a document pair (suffixed with _<language code>)
    """

    lang1, lang1_df, lang2, lang2_df = get_lang1_lang2_dataframes(
        df, languages, min_doc_len, number_to_letter_ratio
    )

    if lang1_df.empty or lang2_df.empty:
        return pd.DataFrame()

    matches = align(
        lang1_documents=lang1_df.fulltext_joined.to_list(),
        lang2_documents=lang2_df.fulltext_joined.to_list(),
        lang1_filename_identifier=f"{website_name}_{lang1}",
        lang2_filename_identifier=f"{website_name}_{lang2}",
        embedding_dir=embedding_dir,
        embedding_model=embedding_model,
        match_threshold=match_threshold,
        aggregation_strategy=aggregation_strategy,
        batch_size=batch_size,
    )

    if matches:
        lang1_indices, lang2_indices = zip(*matches)

        lang1_df = lang1_df.loc[list(lang1_indices)].reset_index(drop=True)
        lang2_df = lang2_df.loc[list(lang2_indices)].reset_index(drop=True)

        df = lang1_df.merge(
            lang2_df, on=lang1_df.index, suffixes=("_" + lang1, "_" + lang2)
        )
        logger.debug("Number of aligned documents: %s", len(df))
        logger.debug(df.columns)
        return df

    return pd.DataFrame()
