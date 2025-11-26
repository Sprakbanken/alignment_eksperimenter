import logging
from datetime import datetime
import sys
from pathlib import Path

from align_documents.utils.config import get_config as get_config
from align_documents.utils.get_embedding_model import (
    get_embedding_model as get_embedding_model,
)

def get_time() -> str:
    return datetime.now().strftime("%Y-%m-%d_%H-%M")

def setup_logging(source_script: str, log_level: str, log_dir: Path = Path("logs")):
    current_time = get_time()
    log_dir.mkdir(parents=True, exist_ok=True)

    # Set up logging
    log_file = log_dir / f"{source_script}_{current_time}.log"
    logging.basicConfig(
        level=getattr(logging, log_level),
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(stream=sys.stdout),
        ],
    )
