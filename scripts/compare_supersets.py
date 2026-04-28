from pathlib import Path
import logging
from align_documents.utils import setup_logging
import pandas as pd

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    setup_logging("compare_supersets", log_level="INFO")

    superset_1 = Path("data/maalfrid_superset")
    superset_2 = Path("data/maalfrid_superset_alt")

    superset_1_files = set(e.name for e in superset_1.iterdir())
    superset_2_files = set(e.name for e in superset_2.iterdir())
    overlap = superset_1_files.intersection(superset_2_files)

    logger.info("len(superset_1_files): %d", len(superset_1_files))
    logger.info("len(superset_2_files): %d", len(superset_2_files))
    logger.info("File overlap: %s", len(overlap))
    logger.info(
        "Files in superset_1 but not in 2: %d", len(superset_1_files - superset_2_files)
    )
    logger.info(
        "Files in superset_2 but not in 1: %d", len(superset_2_files - superset_1_files)
    )

    logger.info("Per-file difference between sets")
    for filename in overlap:
        f1 = superset_1 / filename
        f2 = superset_2 / filename

        doc_hash_1 = pd.read_json(f1, lines=True).doc_hash
        doc_hash_2 = pd.read_json(f2, lines=True).doc_hash
        set_1 = set(doc_hash_1)
        set_2 = set(doc_hash_2)
        if len(set_1.intersection(set_2)) == len(set_1) == len(set_2):
            logger.debug("Filename: %s Supersets are the same", filename)
            continue

        logger.info(
            "Filename: %s len(superset 1): %d len(superset 2): %d Overlap: %d Set 1 only: %d Set 2 only: %d",
            filename,
            len(doc_hash_1),
            len(doc_hash_2),
            len(set_1.intersection(set_2)),
            len(set_1 - set_2),
            len(set_2 - set_1),
        )
