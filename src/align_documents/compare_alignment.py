from align_documents.align import align
from align_documents.utils.config import Config
from align_documents.utils import get_embedding_model
from collections.abc import Iterable
from typing import TypedDict
import logging

logger = logging.getLogger(__name__)


class ConfusionMatrix(TypedDict):
    true_positives: int
    false_negatives: int
    false_positives: int
    true_negatives: int


def compare_alignment(
    config: Config,
    positive_pairs: Iterable[tuple[str, str]],
    filename_prefix: str,
    negative_pairs: Iterable[tuple[str, str]] | None = None,
) -> ConfusionMatrix:
    """Compare our alignment pipeline to the provided reference pairs.
    Count true positives, false negatives, and if negative pairs: true negatives and false positives too.
    """
    lang1, lang2 = config.languages
    lang1_docs, lang2_docs = zip(*positive_pairs)

    embedding_model = get_embedding_model(config.embedding_model)
    matches = align(
        lang1_documents=lang1_docs,
        lang2_documents=lang2_docs,
        lang1_filename_identifier=f"positive_{filename_prefix}_{lang1}",
        lang2_filename_identifier=f"positive_{filename_prefix}_{lang2}",
        embedding_model=embedding_model,
        embedding_dir=config.embedding_dir,
        match_threshold=config.match_threshold,
        batch_size=config.batch_size,
        aggregation_strategy=config.aggregation_strategy,
    )

    true_positives = sum(
        lang1_index == lang2_index for lang1_index, lang2_index in matches
    )
    false_negatives = len(positive_pairs) - true_positives
    logger.debug("True positives: %s", true_positives)
    logger.debug("False negatives: %s", false_negatives)

    if not negative_pairs:
        return ConfusionMatrix(
            true_positives=true_positives, false_negatives=false_negatives
        )

    lang1_docs, lang2_docs = zip(*negative_pairs)

    matches = align(
        lang1_documents=lang1_docs,
        lang2_documents=lang2_docs,
        lang1_filename_identifier=f"negative_{filename_prefix}_{lang1}",
        lang2_filename_identifier=f"negative_{filename_prefix}_{lang2}",
        embedding_model=embedding_model,
        embedding_dir=config.embedding_dir,
        match_threshold=config.match_threshold,
        batch_size=config.batch_size,
        aggregation_strategy=config.aggregation_strategy,
    )

    false_positives = sum(
        lang1_index == lang2_index for lang1_index, lang2_index in matches
    )
    true_negatives = len(negative_pairs) - false_positives
    logger.debug("False positives: %s", false_positives)

    return ConfusionMatrix(
        true_positives=true_positives,
        false_positives=false_positives,
        false_negatives=false_negatives,
        true_negatives=true_negatives,
    )


def calculate_scores(confusion_matrix: ConfusionMatrix) -> dict[str, float]:
    scores = {}
    tp = confusion_matrix["true_positives"]
    fn = confusion_matrix["false_negatives"]
    scores["recall"] = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    if "false_positives" in confusion_matrix:
        fp = confusion_matrix["false_positives"]
        tn = confusion_matrix["true_negatives"]
        scores["precision"] = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        scores["f1"] = (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0
        scores["accuracy"] = (
            (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
        )
    return scores


def log_scores(confusion_matrix: ConfusionMatrix) -> None:
    scores = calculate_scores(confusion_matrix)
    for name, value in scores.items():
        logger.info("%s: %s", name.capitalize(), value)
