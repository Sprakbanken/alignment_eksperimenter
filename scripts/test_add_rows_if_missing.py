import pandas as pd

from make_målfrid_superset_alt import add_rows_if_missing


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
    df = pd.DataFrame()
    df = add_rows_if_missing(df, pd.DataFrame([ROW_A]))
    df = add_rows_if_missing(df, pd.DataFrame([ROW_A]))
    assert len(df) == 1


def test_removes_exact_duplicate_within_other_df():
    df = pd.DataFrame()
    df = add_rows_if_missing(df, pd.DataFrame([ROW_A, ROW_A]))
    assert len(df) == 1


def test_removes_exact_duplicate_when_both_have_it():
    df = add_rows_if_missing(pd.DataFrame([ROW_A, ROW_A]), pd.DataFrame([ROW_A, ROW_A]))
    assert len(df) == 1


def test_keeps_all_unique_rows():
    df = pd.DataFrame()
    df = add_rows_if_missing(df, pd.DataFrame([ROW_A]))
    df = add_rows_if_missing(df, pd.DataFrame([ROW_B]))
    assert len(df) == 2


def test_skips_old_content_for_same_url():
    """Same URL, different hash: keep the newer version only."""
    df = pd.DataFrame()
    df = add_rows_if_missing(df, pd.DataFrame([ROW_A]))
    df = add_rows_if_missing(df, pd.DataFrame([ROW_A_OLD_TEXT]))

    assert len(df) == 1
    assert "A_hash" in set(df.doc_hash)
    assert "A_old_hash" not in set(df.doc_hash)


def test_skips_duplicate_text_at_different_url():
    """Different URL, same hash: don't add a second copy."""
    df = pd.DataFrame()
    df = add_rows_if_missing(df, pd.DataFrame([ROW_B]))
    df = add_rows_if_missing(df, pd.DataFrame([ROW_SAME_TEXT_AS_B]))

    assert len(df) == 1
    assert "url_2" in set(df.url)


def test_skips_outdated_content_when_hash_blocks_url():
    """url_2 had B_hash (2021) then was updated to A_hash (2022).
    A_hash also exists at url_1. ROW_B_URL_A_TEXT is skipped (A_hash exists),
    so url_2 is never claimed, and ROW_B sneaks through."""
    df = pd.DataFrame()
    df = add_rows_if_missing(df, pd.DataFrame([ROW_A]))
    df = add_rows_if_missing(df, pd.DataFrame([ROW_B_URL_A_TEXT_NEWER]))
    result = add_rows_if_missing(df, pd.DataFrame([ROW_B]))

    assert len(result) == 1
    assert "A_hash" in set(result.doc_hash)
    assert "url_2" in set(result.url)
    assert "B_hash" not in set(result.doc_hash), (
        "B_hash is outdated content for url_2 and should not be kept"
    )
