import pandas as pd
from pathlib import Path


def json_data_row_to_csv_row(row: pd.Series) -> pd.Series:
    annotations = {
        e["from_name"]: e["value"]["choices"][0] for e in row.annotations[0]["result"]
    }
    if "sensitive" not in annotations:
        annotations["sensitive"] = False

    annotations["id"] = row.id
    annotations["updated_at"] = row.updated_at
    annotations["created_at"] = row.created_at

    return pd.Series({**row.data, **annotations})


def json_data_to_csv_data(json_df: pd.DataFrame) -> pd.DataFrame:
    return json_df.apply(json_data_row_to_csv_row, axis=1)


def remove_duplicate_annotations(df: pd.DataFrame) -> pd.DataFrame:
    """If there are any duplicate annotations in dataframe, keep the latest updated annotation"""
    only_duplicate_rows = df[
        df.duplicated(subset=["doc_hash_lang_1", "doc_hash_lang_2"], keep=False)
    ]
    if len(only_duplicate_rows) == 0:
        return df

    print("Found duplicate rows in dataframe")
    print("len(df) before deduplication:", len(df))

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
            selected_row = group.sort_values("updated_at", ascending=False).iloc[0]
            print(
                (
                    f"Annotation conflict:\ndoc_hash_lang_1: {doc_hash_lang_1}\ndoc_hash_lang_2: {doc_hash_lang_2}\n"
                    f"Annotations: {group.annotation.to_list()}\n"
                    f"Keeping latest updated annotation: {selected_row.annotation}\n"
                )
            )
            rows_to_keep.append(selected_row)
        else:
            rows_to_keep.append(group.iloc[0])

    print(f"There were {num_duplicated_rows} duplicated rows between the two files")

    rows_to_keep_df = pd.DataFrame(rows_to_keep)
    assert len(rows_to_keep_df) == len(rows_to_keep)

    df = pd.concat((df_with_all_duplicates_removed, rows_to_keep_df), ignore_index=True)
    print(f"len(df) after deduplication: {len(df)}")
    return df


if __name__ == "__main__":
    p1 = Path("manual_annotation/annotated_data/annotator_1/annotated_data_0.json")
    p2 = Path("manual_annotation/annotated_data/annotator_1/annotated_data_1.json")
    p3 = Path("manual_annotation/annotated_data/annotator_1/annotated_data_part_0.json")
    p4 = Path("manual_annotation/annotated_data/annotator_1/annotated_data_part_1.json")

    outfile = Path("manual_annotation/annotated_data/annotator_1.csv")

    df1 = pd.read_json(p1)
    df1 = json_data_to_csv_data(df1)

    df2 = pd.read_json(p2)
    df2 = json_data_to_csv_data(df2)

    df3 = pd.read_json(p3)
    df3 = json_data_to_csv_data(df3)

    df4 = pd.read_json(p4)
    df4 = json_data_to_csv_data(df4)

    print("Json df columns:", df1.columns)

    assert set(df1.columns) == set(df2.columns) == set(df3.columns) == set(df4.columns)

    csv_df = pd.read_csv("manual_annotation/annotated_data/annotator_2.csv")

    csv_cols = set(csv_df.columns)
    json_cols = set(df1.columns)

    # json dfs have no columns not present in csv df
    assert len(json_cols - csv_cols) == 0

    all_df = remove_duplicate_annotations(df=pd.concat([df1, df2, df3, df4]))
    all_df.to_csv(outfile, index=False)
