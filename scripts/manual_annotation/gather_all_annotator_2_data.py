import pandas as pd
from pathlib import Path
from gather_all_annotator_1_data import remove_duplicate_annotations

if __name__ == "__main__":
    p1 = Path("manual_annotation/annotated_data/annotator_2/annotator_2_pre.csv")
    p2 = Path("manual_annotation/annotated_data/annotator_2/annotator_2.csv")

    outfile = Path("manual_annotation/annotated_data/annotator_2.csv")

    df1 = pd.read_csv(p1)
    df2 = pd.read_csv(p2)
    print(f"len(df1): {len(df1)}\nlen(df2):{len(df2)}\n")

    df = remove_duplicate_annotations(pd.concat((df1, df2)))

    annotator_1_df = pd.read_csv("manual_annotation/annotated_data/annotator_1.csv")
    annotator_1_cols = set(annotator_1_df.columns)
    annotator_2_cols = set(df.columns)
    columns_to_drop = annotator_2_cols - annotator_1_cols
    print(
        "Columns in annotator 2 csv files that are not in annotator 1 file (dropping these columns):",
        columns_to_drop,
    )
    df = df[[col for col in df.columns if col not in columns_to_drop]]
    df.to_csv(outfile, index=False)
