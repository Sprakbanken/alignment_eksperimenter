import logging
from pathlib import Path
import tomllib
from dataclasses import dataclass

from align_documents.types import AggregationStrategy
from align_documents.utils.logging import setup_logging
from typing import Self


logger = logging.getLogger(__name__)


@dataclass
class Config:
    data_dir: Path
    embedding_dir: Path
    output_dir: Path
    embedding_model: str
    match_threshold: float
    aggregation_strategy: AggregationStrategy
    batch_size: int
    languages: tuple[str, str]
    number_to_letter_ratio: float
    min_document_length: int

    def validate_and_cast(self):
        self.data_dir = Path(self.data_dir)
        if not self.data_dir.exists():
            raise FileNotFoundError("data_dir does not exist.")

        if len(self.languages) != 2:
            raise ValueError("languages must (only) contain two languages")

        self.embedding_dir = Path(self.embedding_dir)
        self.output_dir = Path(self.output_dir)
        self.aggregation_strategy = AggregationStrategy(self.aggregation_strategy)
        self.languages = tuple(self.languages)

    @classmethod
    def from_dict(cls, config_dict: dict) -> Self:
        config = Config(**config_dict)
        config.validate_and_cast()
        return config


def get_config(config_file: Path) -> Config:
    with open(config_file, "rb") as f:
        config = tomllib.load(f)
    return Config.from_dict(config)


if __name__ == "__main__":
    setup_logging("config", log_level="DEBUG")
    c = get_config(Path("alignment_config.toml"))
    logger.info(c)
