import logging
from pathlib import Path
import tomllib

from align_documents.types import AggregationStrategy, Config

logger = logging.getLogger(__name__)


def validate_config(config: dict) -> Config:
    if config["data_dir"]:
        config["data_dir"] = Path(config["data_dir"])
        if not config["data_dir"].exists():
            raise ValueError("Data directory does not exist.")

    config_keys = [
        "output_dir",
        "embedding_model",
        "embedding_dir",
        "batch_size",
        "aggregation_strategy",
        "match_threshold",
        "languages",
        "number_to_letter_ratio",
        "min_document_length",
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
        "number_to_letter_ratio": lambda x: float(x),
        "min_document_length": lambda x: int(x),
    }

    for key, cast_function in config_casts.items():
        config[key] = cast_function(config[key])

    if len(config["languages"]) != 2:
        raise ValueError("languages must (only) contain two languages")

    return config


def get_config(config_file: Path) -> dict:
    with open(config_file, "rb") as f:
        config = tomllib.load(f)

    logger.info(config)
    return validate_config(config)
