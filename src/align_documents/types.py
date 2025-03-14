from typing import TypedDict


class Match(TypedDict):
    corpus_id: int
    score: float


class AggregationStrategy(str):
    def __new__(cls, value):
        if value not in ["cut-off", "mean"]:
            raise ValueError("Invalid aggregation strategy")
        return str.__new__(cls, value)
