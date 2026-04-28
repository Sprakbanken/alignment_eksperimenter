import pandas as pd

from make_målfrid_superset import dedupe_keep_last


def _make_row(url: str, doc_hash: str, date: str):
    return {
        "url": url,
        "doc_hash": doc_hash,
        "date": date,
        "lang": "nno",
        "fulltext": [f"content for {doc_hash}"],
    }


ROW_A = _make_row("url_1", "A_hash", "2021-06-01")
ROW_B = _make_row("url_2", "B_hash", "2021-06-01")

# Older rows with duplicate contents
ROW_A_OLD_TEXT = _make_row("url_1", "A_old_hash", "2019-06-01")
ROW_SAME_TEXT_AS_B = _make_row("url_3", "B_hash", "2019-06-01")

# Newer row: url_2 was updated to have the same text as ROW_A
ROW_B_URL_A_TEXT_NEWER = _make_row("url_2", "A_hash", "2022-06-01")


def test_removes_exact_duplicate():
    df = dedupe_keep_last(pd.DataFrame([ROW_A, ROW_A]))
    assert len(df) == 1


def test_removes_multiple_exact_duplicates():
    df = dedupe_keep_last(pd.DataFrame([ROW_A, ROW_A, ROW_A, ROW_A]))
    assert len(df) == 1


def test_keeps_all_unique_rows():
    df = pd.DataFrame([ROW_A, ROW_B])
    result = dedupe_keep_last(df)
    assert len(result) == 2


def test_skips_old_content_for_same_url():
    """Same URL, different hash: keep the newer version only."""
    df = pd.DataFrame([ROW_A, ROW_A_OLD_TEXT])
    result = dedupe_keep_last(df)

    assert len(result) == 1
    assert "A_hash" in set(result.doc_hash)
    assert "A_old_hash" not in set(result.doc_hash)


def test_skips_duplicate_text_at_different_url():
    """Different URL, same hash: keep only one copy."""
    df = pd.DataFrame([ROW_B, ROW_SAME_TEXT_AS_B])
    result = dedupe_keep_last(df)

    assert len(result) == 1
    assert "url_2" in set(result.url)


def test_skips_outdated_content_when_hash_blocks_url():
    """url_2 had B_hash (2021) then was updated to A_hash (2022).
    A_hash also exists at url_1. URL dedup removes ROW_B (keeping
    ROW_B_URL_A_TEXT), then hash dedup removes ROW_B_URL_A_TEXT
    (keeping ROW_A). B_hash is correctly gone."""
    df = pd.DataFrame([ROW_A, ROW_B_URL_A_TEXT_NEWER, ROW_B])
    result = dedupe_keep_last(df)

    assert len(result) == 1
    assert "A_hash" in set(result.doc_hash)
    assert "url_2" in set(result.url)
    assert "B_hash" not in set(result.doc_hash), (
        "B_hash is outdated content for url_2 and should not be kept"
    )
