import pandas as pd
from pathlib import Path


def json_data_row_to_csv_row(row: pd.Series) -> pd.Series:
    annotations = {
        e["from_name"]: e["value"]["choices"][0] for e in row.annotations[0]["result"]
    }
    # print(row.index)
    #   Index(['id', 'annotations', 'file_upload', 'drafts', 'predictions', 'data',
    #    'meta', 'created_at', 'updated_at', 'inner_id', 'total_annotations',
    #    'cancelled_annotations', 'total_predictions', 'comment_count',
    #    'unresolved_comment_count', 'last_comment_updated_at', 'project',
    #    'updated_by', 'comment_authors'],

    if "sensitive" not in annotations:
        annotations["sensitive"] = False

    return pd.Series({**row.data, **annotations})


def json_data_to_csv_data(json_df: pd.DataFrame) -> pd.DataFrame:
    return json_df.apply(json_data_row_to_csv_row, axis=1)


if __name__ == "__main__":
    p1 = Path("manual_annotation/annotated_data/annotator_1/annotated_data_0.json")
    p2 = Path("manual_annotation/annotated_data/annotator_1/annotated_data_1.json")
    p3 = Path("manual_annotation/annotated_data/annotator_1/annotated_data_part_0.json")
    p4 = Path("manual_annotation/annotated_data/annotator_1/annotated_data_part_1.json")

    json_df1 = pd.read_json(p1)
    json_df1 = json_data_to_csv_data(json_df1)

    json_df2 = pd.read_json(p2)
    json_df2 = json_data_to_csv_data(json_df2)

    json_df3 = pd.read_json(p3)
    json_df3 = json_data_to_csv_data(json_df3)

    json_df4 = pd.read_json(p4)
    json_df4 = json_data_to_csv_data(json_df4)

    print("Json df columns")
    print(json_df1.columns)

    assert (
        set(json_df1.columns)
        == set(json_df2.columns)
        == set(json_df3.columns)
        == set(json_df4.columns)
    )
