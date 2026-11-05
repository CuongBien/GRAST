"""Strict numeric-v1 extraction and exact rational comparison; no float tolerance."""

from __future__ import annotations

import re
import unicodedata
from decimal import Decimal, InvalidOperation
from fractions import Fraction

_NUMBER = re.compile(r"[+-]?(?:(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d{1,3})?\Z")


def _decimal(value: str) -> Fraction | None:
    if not _NUMBER.fullmatch(value):
        return None
    try:
        decimal = Decimal(value.replace(",", ""))
        if not decimal.is_finite() or abs(decimal.adjusted()) > 100:
            return None
        return Fraction(decimal)
    except (InvalidOperation, ValueError, OverflowError):
        return None


def normalize_answer(text: str) -> str | None:
    """Accept one number/fraction, optional dollar/boxed wrapper, NFKC and Unicode minus.

    Units, percentages, multiple numbers, NaN, infinity and arithmetic expressions are invalid.
    The returned canonical value is an integer or a reduced numerator/denominator string.
    """
    if not isinstance(text, str) or len(text) > 256:
        return None
    value = unicodedata.normalize("NFKC", text).strip().replace("−", "-").replace("⁄", "/")
    boxed = re.fullmatch(r"\\boxed\{(.+)\}", value)
    if boxed:
        value = boxed[1].strip()
    latex = re.fullmatch(r"([+-]?)\\frac\{([+-]?\d+)\}\{([+-]?\d+)\}", value)
    if latex:
        numerator = int(latex[2]) * (-1 if latex[1] == "-" else 1)
        value = f"{numerator}/{latex[3]}"
    value = re.sub(r"^([+-]?)\$\s*", r"\1", value)
    if "/" in value:
        parts = value.split("/")
        if len(parts) != 2:
            return None
        numerator, denominator = (_decimal(part.strip()) for part in parts)
        if numerator is None or denominator in (None, 0):
            return None
        number = numerator / denominator
    else:
        number = _decimal(value)
    if number is None:
        return None
    return str(number.numerator) if number.denominator == 1 else str(number)


def extract_answer(text: str) -> str | None:
    """Extract one explicit <final> block or #### suffix, otherwise a bare numeric answer.

    Malformed/repeated markers fail closed; numbers in a rationale are never guessed.
    """
    if not isinstance(text, str) or len(text) > 65536:
        return None
    if "<final" in text or "</final" in text:
        matches = re.findall(r"<final>(.*?)</final>", text, flags=re.DOTALL)
        if len(matches) != 1 or text.count("<final") != 1 or text.count("</final") != 1:
            return None
        # An explicit final block is authoritative; trailing prose is not another answer.
        return normalize_answer(matches[0])
    if "####" in text:
        if text.count("####") != 1:
            return None
        return normalize_answer(text.split("####", 1)[1].strip())
    return normalize_answer(text)


def answers_equal(prediction: str, reference: str) -> bool:
    predicted, expected = extract_answer(prediction), extract_answer(reference)
    return predicted is not None and expected is not None and predicted == expected
