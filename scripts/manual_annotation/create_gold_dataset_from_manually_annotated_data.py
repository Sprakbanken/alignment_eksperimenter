import logging
from pathlib import Path

import pandas as pd
from align_documents.utils import setup_logging
from align_documents.utils.dataframe import add_pair_key

logger = logging.getLogger(__name__)


def filter_rows(df: pd.DataFrame) -> pd.DataFrame:
    df = df[df["annotation"] != "Something is wrong"]
    df = df[df["sensitive"] != "Example contains personal info"]
    return df


if __name__ == "__main__":
    setup_logging("create_gold_dataset_from_manually_annotated_data", "INFO")

    annotated_data_dir = Path("manual_annotation/annotated_data")
    outfile = annotated_data_dir / "gold_data.csv"

    gold_columns = [
        "doc_hash_lang_1",
        "lang_lang_1",
        "url_lang_1",
        "domain_lang_1",
        "date_lang_1",
        "mimetype_lang_1",
        "fulltext_lang_1",
        "fulltext_joined_lang_1",
        "doc_hash_lang_2",
        "lang_lang_2",
        "url_lang_2",
        "domain_lang_2",
        "date_lang_2",
        "mimetype_lang_2",
        "fulltext_lang_2",
        "fulltext_joined_lang_2",
        "annotation",
        "sensitive",
    ]

    a1 = pd.read_csv(annotated_data_dir / "annotator_1.csv")
    a2 = pd.read_csv(annotated_data_dir / "annotator_2.csv")
    resolved = pd.read_csv(annotated_data_dir / "resolved_conflicting_annotations.csv")

    a1 = add_pair_key(a1)
    a2 = add_pair_key(a2)
    resolved = add_pair_key(resolved)

    resolved_keys = set(resolved["pair_key"])
    a1_keys = set(a1["pair_key"])
    a2_keys = set(a2["pair_key"])

    # Pairs annotated only by annotator 1
    a1_only = a1[~a1["pair_key"].isin(a2_keys)]

    # Pairs annotated only by annotator 2
    a2_only = a2[~a2["pair_key"].isin(a1_keys)]

    # Pairs annotated by both and in agreement (not in resolved conflicts)
    agreed = a1[a1["pair_key"].isin(a2_keys) & ~a1["pair_key"].isin(resolved_keys)]

    logger.info(f"a1-only pairs:  {len(a1_only)}")
    logger.info(f"a2-only pairs:  {len(a2_only)}")
    logger.info(f"Agreed pairs:   {len(agreed)}")
    logger.info(f"Resolved pairs: {len(resolved)}")

    gold = pd.concat(
        [
            a1_only[gold_columns],
            a2_only[gold_columns],
            agreed[gold_columns],
            resolved[gold_columns],
        ],
        ignore_index=True,
    )

    logger.info(f"Total before filtering: {len(gold)}")
    gold = filter_rows(gold)
    logger.info(f"Total after filtering:  {len(gold)}")

    gold.to_csv(outfile, index=False)
    logger.info(f"Saved gold dataset to {outfile}")

    logger.info("Doc pairs per language pair:")
    lang_pair_counts = (
        gold.groupby(["lang_lang_1", "lang_lang_2"]).size().rename("total")
    )
    annotation_counts = (
        gold.groupby(["lang_lang_1", "lang_lang_2", "annotation"])
        .size()
        .unstack(fill_value=0)
    )
    stats = annotation_counts.join(lang_pair_counts)
    for (lang_1, lang_2), row in stats.iterrows():
        annotation_str = ", ".join(
            f"{col}: {row[col]}" for col in annotation_counts.columns
        )
        logger.info(f"  {lang_1}-{lang_2}: total={row['total']} ({annotation_str})")
