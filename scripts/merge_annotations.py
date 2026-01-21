import pandas as pd
from pathlib import Path


if __name__ == "__main__":
    annotations_1 = Path("manual_annotation/annotated_data/annotator_2.csv")
    annotations_2 = Path(
        "manual_annotation/annotated_data/project-2-at-2026-01-21-12-12-5e42b310.csv"
    )
    output_p = Path("manual_annotation/annotated_data/annotator_2.csv")

    df1 = pd.read_csv(annotations_1)
    df2 = pd.read_csv(annotations_2)

    print(len(df1))
    print(len(df2))

    assert set(df1.columns) == set(df2.columns)

    print(df1.columns)

    df = pd.concat([df1, df2])
    print(f"len(df) before deduplication: {len(df)}")

    # Get all lines where there are duplicates
    only_duplicate_rows = df[
        df.duplicated(subset=["doc_hash_lang_1", "doc_hash_lang_2"], keep=False)
    ]
    df_with_all_duplicates_removed = df.drop_duplicates(
        subset=["doc_hash_lang_1", "doc_hash_lang_2"], keep=False
    )

    num_duplicated_rows = 0
    rows_to_keep = []
    for (doc_hash_lang_1, doc_hash_lang_2), group in only_duplicate_rows.groupby(
        ["doc_hash_lang_1", "doc_hash_lang_2"]
    ):
        num_duplicated_rows += 1
        if len(group.annotation.unique()) > 1:
            selected_row = group.loc[group.updated_at.idxmax()]
            print(
                (
                    f"Annotation conflict:\ndoc_hash_lang_1: {doc_hash_lang_1}\ndoc_hash_lang_2: {doc_hash_lang_2}\n"
                    f"Annotions: {group.annotation.to_list()}\n"
                    f"Keeping latest updated annotation: {selected_row.annotation}\n"
                )
            )
            rows_to_keep.append(selected_row)
        else:
            first_row_index = group.index[0]
            rows_to_keep.append(group.loc[first_row_index])

    print(f"There were {num_duplicated_rows} duplicated rows between the two files")
    rows_to_keep_df = pd.DataFrame(rows_to_keep)
    assert len(rows_to_keep_df) == len(rows_to_keep)

    df = pd.concat((df_with_all_duplicates_removed, rows_to_keep_df), ignore_index=True)
    print(f"len(df) after deduplication: {len(df)}")

    df.to_csv(output_p, index=False)
    print(f"Saved merged and deduplicated df to {output_p}")
