"""
Universal Unicode text normalization, accent stripping, corporate suffix filtering,
and address standardization. 100% country-agnostic.
"""

import re
import unicodedata
from typing import List, Optional, Set

# Regex Patterns
RE_NON_ALPHANUM = re.compile(r"[^\w\s]")
RE_EXTRA_SPACES = re.compile(r"\s+")

# Comprehensive Multi-Lingual Corporate Suffixes (US, UK, India, France, Germany, etc.)
CORPORATE_SUFFIXES: Set[str] = {
    # US / UK / International
    "inc", "incorporated", "corp", "corporation", "llc", "llp", "lp", "ltd", "limited",
    "co", "company", "pllc", "holdings", "group", "services", "enterprises", "solutions",
    "industries", "associates", "partners", "ventures", "technologies", "international",
    # India
    "pvt", "private", "proprietorship", "traders", "trading", "sangh",
    # France / Romance
    "sarl", "sasu", "sas", "eurl", "sa", "sci", "snc", "gie", "fils", "et fils", "cie", "ste",
    # Germany / Central Europe
    "gmbh", "ag", "ug", "kg", "kgaa", "holding",
}

# Standard Address Term Standardizations
ADDRESS_ABBREVIATIONS = {
    "rd": "road",
    "st": "street",
    "ave": "avenue",
    "blvd": "boulevard",
    "dr": "drive",
    "ln": "lane",
    "ct": "court",
    "pl": "place",
    "pkwy": "parkway",
    "apt": "apartment",
    "ste": "suite",
    "bldg": "building",
    "fl": "floor",
    "po box": "pobox",
    "p o box": "pobox",
    "post office box": "pobox",
    "hwy": "highway",
    "nr": "near",
    "opp": "opposite",
    "bd": "boulevard",
    "rte": "route",
    "av": "avenue",
}


def strip_accents(text: str) -> str:
    """
    Strips accents and diacritics using Unicode NFD normalization.
    Converts 'Président' -> 'President', 'Bordeaux' -> 'Bordeaux', 'élève' -> 'eleve'.
    """
    if not text or not isinstance(text, str):
        return ""
    nfkd = unicodedata.normalize("NFKD", text)
    return "".join(c for c in nfkd if not unicodedata.combining(c))


def clean_text(text: str) -> str:
    """
    Basic universal text cleaning: lowercase, accent stripping, punctuation normalization.
    """
    if not text or not isinstance(text, str) or text == "nan":
        return ""
    text = strip_accents(text.lower())
    text = text.replace("&", " and ")
    text = RE_NON_ALPHANUM.sub(" ", text)
    return RE_EXTRA_SPACES.sub(" ", text).strip()


def clean_business_name(name: str, strip_legal_suffixes: bool = True) -> str:
    """
    Cleans business name and optionally removes corporate legal suffixes.
    """
    cleaned = clean_text(name)
    if not cleaned:
        return ""

    if strip_legal_suffixes:
        tokens = cleaned.split()
        filtered = [t for t in tokens if t not in CORPORATE_SUFFIXES]
        if filtered:
            return " ".join(filtered)
    return cleaned


def clean_address(address: str, expand_abbrevs: bool = True) -> str:
    """
    Normalizes business address and standardizes common road/street abbreviations.
    """
    cleaned = clean_text(address)
    if not cleaned:
        return ""

    if expand_abbrevs:
        tokens = cleaned.split()
        expanded = [ADDRESS_ABBREVIATIONS.get(t, t) for t in tokens]
        cleaned = " ".join(expanded)

    return cleaned
