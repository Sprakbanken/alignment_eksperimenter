import pandas as pd
import logging
from align_documents.utils import setup_logging


logger = logging.getLogger(__name__)


def json_data_row_to_csv_row(row: pd.Series) -> pd.Series:
    annotations = {
        e["from_name"]: e["value"]["choices"][0] for e in row.annotations[0]["result"]
    }
    if "sensitive" not in annotations:
        annotations["sensitive"] = False

    return pd.Series({**row.data, **annotations})


def json_data_to_csv_data(json_df: pd.DataFrame) -> pd.DataFrame:
    return json_df.apply(json_data_row_to_csv_row, axis=1)


if __name__ == "__main__":
    setup_logging("merge_annotation_jsons", "DEBUG")

    json_1 = "manual_annotation/annotated_data/annotated_data_0.json"
    json_2 = "manual_annotation/annotated_data/annotated_data_1.json"
    output = "manual_annotation/annotated_data/annotator_1.csv"

    logger.info(f"Reading JSON files: {json_1}, {json_2}")
    df1 = pd.read_json(json_1)
    df2 = pd.read_json(json_2)

    df1 = json_data_to_csv_data(df1)
    df2 = json_data_to_csv_data(df2)

    df = pd.concat([df1, df2], ignore_index=True)
    df.to_csv(output, index=False)

    cols_to_keep = df.columns

    other_csv = "manual_annotation/annotated_data/annotator_2.csv"
    df = pd.read_csv(other_csv)
    df = df[cols_to_keep]
    df.to_csv(other_csv, index=False)
