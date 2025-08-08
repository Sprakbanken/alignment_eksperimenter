from typing import TypedDict
from pathlib import Path


class Match(TypedDict):
    corpus_id: int
    score: float


class AggregationStrategy(str):
    def __new__(cls, value):
        if value not in ["cut-off", "mean"]:
            raise ValueError("Invalid aggregation strategy")
        return str.__new__(cls, value)


class Config(TypedDict):
    embedding_dir: Path
    output_dir: Path
    match_threshold: float
    aggregation_strategy: AggregationStrategy
    batch_size: int
    languages: tuple
    number_to_letter_ratio: float
    min_document_length: int
