import pandas as pd


def concat_hashes(hash1: str, hash2: str) -> str:
    return hash1 + hash2


if __name__ == "__main__":
    df = pd.read_json(
        "data/output/data_for_manual_annotation/data_to_annotate.jsonl", lines=True
    )
    print(len(df))

    df["doc_hash"] = df.apply(
        lambda row: concat_hashes(row["doc_hash_lang_1"], row["doc_hash_lang_2"]),
        axis=1,
    )
    df["langs"] = df.apply(
        lambda row: f"{row['lang_lang_1']}_{row['lang_lang_2']}", axis=1
    )

    df = df.drop_duplicates(subset=["doc_hash", "langs"])
    print(len(df))
