import argparse
import logging
from pathlib import Path
from align_documents.utils import setup_logging
from sklearn.metrics import cohen_kappa_score
import pandas as pd


logger = logging.getLogger(__name__)


def add_lang1_lang2_hash(df: pd.DataFrame) -> pd.DataFrame:
    df["doc_hash"] = df.doc_hash_lang_1 + df.doc_hash_lang_2
    return df


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Calculate inter-annotator agreement for manual annotations"
    )
    parser.add_argument(
        "--annotator_one",
        type=Path,
        default=Path("manual_annotation/annotated_data/annotator_1.csv"),
        help="Path to the annotated data (.csv-file) from annotator 1 (default: %(default)s)",
    )
    parser.add_argument(
        "--annotator_two",
        type=Path,
        default=Path("manual_annotation/annotated_data/annotator_2.csv"),
        help="Path to the annotated data (.csv-file) from annotator 2 (default: %(default)s)",
    )
    parser.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level (default: %(default)s)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    setup_logging("calculate_annotator_agreement", args.log_level)

    df1 = pd.read_csv(args.annotator_one)
    df2 = pd.read_csv(args.annotator_two)
    logger.debug("len(df1): %s" % len(df1))
    logger.debug("len(df2): %s" % len(df2))

    labels = df1.annotation.unique()
    logger.debug("labels: %s", labels)

    # Concatenate doc hashes for both documents in annotated doc pair
    df1 = add_lang1_lang2_hash(df1)
    df2 = add_lang1_lang2_hash(df2)

    # Deduplicate
    df1 = df1.drop_duplicates("doc_hash")
    logger.debug("len(df1) after deduplication: %s", len(df1))
    df2 = df2.drop_duplicates("doc_hash")
    logger.debug("len(df2) after deduplication: %s", len(df2))

    # Only keep document pairs that are annotated by both annotators
    overlapping_doc_hashes = set(df1.doc_hash).intersection(set(df2.doc_hash))
    logger.info("%s document pairs are doubly annotated", len(overlapping_doc_hashes))

    df1 = df1[df1.doc_hash.isin(overlapping_doc_hashes)]
    logger.debug("len(df1) %s", len(df1))
    df2 = df2[df2.doc_hash.isin(overlapping_doc_hashes)]
    logger.debug("len(df2) %s", len(df2))

    ## TODO: sørg for at df1 og df2 har lik rekkefølge.
    df1 = df1.sort_values(by="doc_hash")
    df1 = df1.set_index("doc_hash")

    df2 = df2.sort_values(by="doc_hash")
    df2 = df2.set_index("doc_hash")

    assert all(df1.index == df2.index)
    logger.debug("df1 and df2 indices are the same")

    score = cohen_kappa_score(y1=df1.annotation, y2=df2.annotation, labels=labels)

    logger.info("Cohens kappa: %s", score)


if __name__ == "__main__":
    main()
