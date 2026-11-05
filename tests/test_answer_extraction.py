"""Numeric-v1 accepts exact equivalent representations and rejects ambiguous answers."""

import pytest

from llm_scratch.reasoning.answer_extraction import (
    answers_equal,
    extract_answer,
    normalize_answer,
)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("42", "42"),
        ("-0.000", "0"),
        ("+0.5", "1/2"),
        ("1,234.50", "2469/2"),
        ("$-1,200.00", "-1200"),
        ("-$2", "-2"),
        ("−３", "-3"),
        ("½", "1/2"),
        (" 2 / -4 ", "-1/2"),
        ("1e3", "1000"),
        ("2.5e-2", "1/40"),
        (".125", "1/8"),
        (r"\boxed{\frac{1}{2}}", "1/2"),
        (r"-\frac{1}{2}", "-1/2"),
        ("100000000000000000001", "100000000000000000001"),
    ],
)
def test_canonical_numbers(raw, expected):
    assert normalize_answer(raw) == expected


@pytest.mark.parametrize(
    "raw",
    [
        "",
        "NaN",
        "inf",
        "1/0",
        "50%",
        "42 apples",
        "1,2",
        "1 2",
        "1+2",
        "1.2.3",
        "1e999",
        "3/4/5",
        "$1$2",
        "one",
        "9" * 300,
        None,
        True,
    ],
)
def test_invalid_numbers_are_not_guessed(raw):
    assert normalize_answer(raw) is None


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("There are 12 objects and another 5.\n#### 17", "17"),
        ("<think>3 or 9 might be plausible.</think><final>2/4</final>", "1/2"),
        ("0.5000", "1/2"),
        ("Calculation goes here.\n#### $1,200", "1200"),
        ("<final>-3</final>", "-3"),
    ],
)
def test_explicit_final_answer(text, expected):
    assert extract_answer(text) == expected


@pytest.mark.parametrize(
    "text",
    [
        "I computed 3 and then 4.",
        "<final>7",
        "<final>1</final><final>2</final>",
        "#### 1\n#### 2",
        "<final>not a number</final>",
        "#### 3 apples",
        "<final>1</final><final broken",
        "<final>1</final></final broken",
        "The answer could be 4 or 5.",
        "",
    ],
)
def test_missing_or_ambiguous_final_answer_fails_closed(text):
    assert extract_answer(text) is None


def test_exact_comparison_does_not_equate_invalids_or_round_numbers():
    assert answers_equal("<final>0.50</final>", "#### 1/2")
    assert not answers_equal("0.3333333333", "1/3")
    assert not answers_equal("bad", "also bad")
    assert not answers_equal("100000000000000000001", "100000000000000000000")
