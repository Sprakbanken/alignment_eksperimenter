from align_documents.utils.dataframe import has_bad_quality


def test_good_doc_has_good_quality():
    good_doc = """
    this document is very good. i give it a 5 star rating.
    """
    assert not has_bad_quality(good_doc, min_len=3, number_to_letter_ratio=0.5)


def test_short_doc_has_bad_quality():
    short_doc = "h"
    assert has_bad_quality(short_doc, min_len=3, number_to_letter_ratio=0.5)


def test_short_doc_has_good_quality_when_short_min_len():
    short_doc = "h"
    assert not has_bad_quality(short_doc, min_len=1, number_to_letter_ratio=0.5)


def test_doc_with_only_numbers_has_bad_quality():
    numbers_doc = "1230"
    assert has_bad_quality(numbers_doc, number_to_letter_ratio=12345, min_len=3)


def test_doc_with_too_many_numbers_has_bad_quality():
    numbers_doc = """
    0+823 81281 828320 3291 
    1230+2 943483 823 72 32543
    we need at least one letter
    """
    assert has_bad_quality(numbers_doc, number_to_letter_ratio=0.5, min_len=3)


def test_doc_with_many_numbers_has_good_quality_when_high_number_to_letter_ratio():
    numbers_doc = """
    0+823 81281 828320 3291 
    1230+2 943483 823 72 32543
    we need at least one letter
    """
    assert not has_bad_quality(numbers_doc, number_to_letter_ratio=100_000, min_len=3)
