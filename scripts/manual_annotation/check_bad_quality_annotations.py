import pandas as pd

if __name__ == "__main__":
    df1 = pd.read_csv("manual_annotation/annotated_data/annotator_1.csv")
    df2 = pd.read_csv("manual_annotation/annotated_data/annotator_2.csv")

    print(df1.annotation.value_counts())
    print(df2.annotation.value_counts())
