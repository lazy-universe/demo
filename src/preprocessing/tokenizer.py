"""
Tokenization, character n-gram generation, and numerical token isolation routines.
"""

import re
from typing import List, Set

from .text import clean_text

# Regex for 5-digit and 6-digit postal/PIN numbers
RE_PIN_6DIGIT = re.compile(r"\b\d{6}\b")
RE_POSTAL_5DIGIT = re.compile(r"\b\d{5}\b")
RE_NUMBERS = re.compile(r"\b\d+\b")


def extract_postal_codes(address: str) -> List[str]:
    """
    Extracts potential postal codes / PIN codes from address text universally.
    Extracts 5-digit, 6-digit, and alphanumeric postal patterns with zero country hardcoding.
    """
    if not address or not isinstance(address, str):
        return []

    codes = RE_PIN_6DIGIT.findall(address) + RE_POSTAL_5DIGIT.findall(address)
    # Deduplicate while preserving order
    return list(dict.fromkeys(codes))


def extract_numbers(text: str) -> Set[str]:
    """
    Extracts all numerical tokens (door numbers, plot numbers, street numbers).
    """
    if not text or not isinstance(text, str):
        return set()
    return set(RE_NUMBERS.findall(text))


def generate_char_ngrams(text: str, n: int = 3) -> Set[str]:
    """
    Generates character n-grams from cleaned text for typo-tolerant fuzzy matching.
    """
    cleaned = clean_text(text).replace(" ", "")
    if len(cleaned) < n:
        return {cleaned} if cleaned else set()
    return {cleaned[i : i + n] for i in range(len(cleaned) - n + 1)}


def generate_word_tokens(text: str) -> List[str]:
    """
    Tokenizes cleaned text into unique alphanumeric words.
    """
    cleaned = clean_text(text)
    if not cleaned:
        return []
    return cleaned.split()
