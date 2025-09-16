import logging
from datetime import datetime
import sys
from pathlib import Path


def setup_logging(source_script: str, log_level: str, log_dir: Path = Path("logs")):
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M")
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
