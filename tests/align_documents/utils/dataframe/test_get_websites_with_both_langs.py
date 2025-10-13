from align_documents.utils.dataframe import get_websites_with_both_langs
import pandas as pd


def test_returns_empty_df_on_empty_input_df():
    df = pd.DataFrame(columns=["website", "language"])
    result = get_websites_with_both_langs(df, languages=("eng", "nob"))
    expected = pd.DataFrame(columns=["website", "language"])
    pd.testing.assert_frame_equal(result, expected)


def test_returns_empty_df_when_no_websites_with_both_langs():
    data = {
        "website": ["site1", "site2"],
        "language": [
            "eng",
            "nob",
        ],
    }
    df = pd.DataFrame(data)
    result = get_websites_with_both_langs(df, languages=("eng", "nob"))
    expected = pd.DataFrame(columns=["website", "language"])
    pd.testing.assert_frame_equal(result, expected)


def test_returns_only_websites_with_both_langs():
    data = {
        "website": ["both_langs", "both_langs", "only_eng", "only_nob"],
        "language": ["eng", "nob", "eng", "nob"],
    }
    df = pd.DataFrame(data)
    result = get_websites_with_both_langs(df, languages=("eng", "nob"))
    expected_data = {
        "website": ["both_langs", "both_langs"],
        "language": ["eng", "nob"],
    }
    expected = pd.DataFrame(expected_data).reset_index(drop=True)
    pd.testing.assert_frame_equal(result.reset_index(drop=True), expected)


def test_removes_rows_in_other_languages():
    data = {
        "website": ["both_langs", "both_langs", "both_langs", "only_eng", "only_nob"],
        "language": ["eng", "nob", "nno", "eng", "nob"],
    }
    df = pd.DataFrame(data)
    result = get_websites_with_both_langs(df, languages=("eng", "nob"))
    expected_data = {
        "website": [
            "both_langs",
            "both_langs",
        ],
        "language": ["eng", "nob"],
    }
    expected = pd.DataFrame(expected_data).reset_index(drop=True)
    pd.testing.assert_frame_equal(result.reset_index(drop=True), expected)
