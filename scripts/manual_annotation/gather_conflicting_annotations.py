import argparse
import logging
from pathlib import Path
from align_documents.utils import setup_logging
from align_documents.utils.dataframe import add_pair_key
import pandas as pd


logger = logging.getLogger(__name__)


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
        "--output_file",
        "-o",
        type=Path,
        default=Path(
            "data/output/data_for_manual_annotation/conflicting_annotation.csv"
        ),
        help="Path to output file to store conflicting annotations",
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
    setup_logging("gather_conflicting_annotations", args.log_level)

    df1 = pd.read_csv(args.annotator_one)
    df2 = pd.read_csv(args.annotator_two)
    logger.debug("len(df1): %s" % len(df1))
    logger.debug("len(df2): %s" % len(df2))

    labels = df1.annotation.unique()
    logger.debug("labels: %s", labels)

    # Build a composite key for both documents in annotated doc pair
    df1 = add_pair_key(df1)
    df2 = add_pair_key(df2)

    # Deduplicate
    df1 = df1.drop_duplicates("pair_key")
    logger.debug("len(df1) after deduplication: %s", len(df1))
    df2 = df2.drop_duplicates("pair_key")
    logger.debug("len(df2) after deduplication: %s", len(df2))

    # Only keep document pairs that are annotated by both annotators
    overlapping_pair_keys = set(df1.pair_key).intersection(set(df2.pair_key))
    logger.info("%s document pairs are doubly annotated", len(overlapping_pair_keys))

    df1 = df1[df1.pair_key.isin(overlapping_pair_keys)]
    df2 = df2[df2.pair_key.isin(overlapping_pair_keys)]

    df1 = df1.sort_values(by="pair_key")
    df1.reset_index(inplace=True)

    df2 = df2.sort_values(by="pair_key")
    df2.reset_index(inplace=True)

    assert all(df1.index == df2.index)
    assert all(df1.pair_key == df2.pair_key)

    conflicting_annotation_mask = df1.annotation != df2.annotation

    df = df1[conflicting_annotation_mask]

    logger.info("%s document pairs have conflicting annotations", len(df))

    df = df.rename(columns={"annotation": "annotator_1_annotation"})
    df["annotator_2_annotation"] = df2[conflicting_annotation_mask].annotation

    for tup in df.itertuples():
        logger.debug(
            "Annotator 1: %s Annotator 2: %s",
            tup.annotator_1_annotation,
            tup.annotator_2_annotation,
        )

    df.to_csv(args.output_file, index=False)


if __name__ == "__main__":
    main()
