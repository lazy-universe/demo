"""
Pairwise Feature Engineering Module for Business Entity Resolution.
Computes string similarities, token overlaps, edit distances, and domain-specific signals.
"""

from typing import Dict, List, Optional, Set, Tuple
import numpy as np
import pandas as pd
from rapidfuzz import fuzz

from ..preprocessing.text import (
    clean_address,
    clean_business_name,
    clean_text,
)
from ..preprocessing.tokenizer import (
    extract_numbers,
    extract_postal_codes,
    generate_char_ngrams,
    generate_word_tokens,
)

FEATURE_NAMES = [
    "name_fuzz_ratio",
    "name_token_sort",
    "name_token_set",
    "name_ngram_jaccard",
    "name_word_jaccard",
    "name_exact_match",
    "name_len_diff",
    "name_len_ratio",
    "addr_token_sort",
    "addr_token_set",
    "addr_word_jaccard",
    "addr_num_jaccard",
    "addr_num_overlap",
    "postal_code_match",
    "is_addr_missing",
    "is_source2",
]


def compute_jaccard_similarity(set_a: Set[str], set_b: Set[str]) -> float:
    """Computes Jaccard similarity between two sets."""
    if not set_a and not set_b:
        return 1.0
    if not set_a or not set_b:
        return 0.0
    intersection = len(set_a.intersection(set_b))
    union = len(set_a.union(set_b))
    return intersection / union if union > 0 else 0.0


def extract_pair_features(
    name1: str,
    addr1: str,
    country1: str,
    name2: str,
    addr2: str,
    country2: str,
    source2_prefix: str = "S2",
) -> Dict[str, float]:
    """
    Extracts numerical similarity features for a single candidate pair (S1, S2/S3).
    """
    # 1. Cleaned texts
    c_name1 = clean_business_name(name1)
    c_name2 = clean_business_name(name2)
    c_addr1 = clean_address(addr1)
    c_addr2 = clean_address(addr2)

    # 2. Name Similarities
    name_fuzz_ratio = fuzz.ratio(c_name1, c_name2) / 100.0
    name_token_sort = fuzz.token_sort_ratio(c_name1, c_name2) / 100.0
    name_token_set = fuzz.token_set_ratio(c_name1, c_name2) / 100.0

    # 3. Name N-gram & Token Jaccard
    name_ngrams1 = generate_char_ngrams(c_name1, n=3)
    name_ngrams2 = generate_char_ngrams(c_name2, n=3)
    name_ngram_jaccard = compute_jaccard_similarity(name_ngrams1, name_ngrams2)

    name_words1 = set(generate_word_tokens(c_name1))
    name_words2 = set(generate_word_tokens(c_name2))
    name_word_jaccard = compute_jaccard_similarity(name_words1, name_words2)

    name_exact_match = 1.0 if c_name1 and c_name1 == c_name2 else 0.0

    # 4. Length Ratios
    len1 = len(c_name1)
    len2 = len(c_name2)
    name_len_diff = abs(len1 - len2)
    name_len_ratio = min(len1, len2) / max(len1, len2) if max(len1, len2) > 0 else 1.0

    # 5. Address Similarities
    is_addr_missing = 1.0 if not c_addr2 or c_addr2 == "nan" else 0.0
    if not is_addr_missing and c_addr1:
        addr_token_sort = fuzz.token_sort_ratio(c_addr1, c_addr2) / 100.0
        addr_token_set = fuzz.token_set_ratio(c_addr1, c_addr2) / 100.0

        addr_words1 = set(generate_word_tokens(c_addr1))
        addr_words2 = set(generate_word_tokens(c_addr2))
        addr_word_jaccard = compute_jaccard_similarity(addr_words1, addr_words2)

        nums1 = extract_numbers(c_addr1)
        nums2 = extract_numbers(c_addr2)
        num_jaccard = compute_jaccard_similarity(nums1, nums2)
        num_overlap_count = float(len(nums1.intersection(nums2)))

        pc1 = set(extract_postal_codes(c_addr1))
        pc2 = set(extract_postal_codes(c_addr2))
        postal_code_match = (
            1.0 if (pc1 and pc2 and len(pc1.intersection(pc2)) > 0) else 0.0
        )
    else:
        addr_token_sort = 0.0
        addr_token_set = 0.0
        addr_word_jaccard = 0.0
        num_jaccard = 0.0
        num_overlap_count = 0.0
        postal_code_match = 0.0

    # 6. Source Flag
    is_source2 = 1.0 if source2_prefix.startswith("S2") else 0.0

    # 7. Composite heuristic score
    if is_addr_missing:
        composite_score = 0.9 * name_token_set + 0.1 * name_token_sort
    else:
        composite_score = (
            0.45 * name_token_set
            + 0.20 * name_ngram_jaccard
            + 0.25 * addr_token_set
            + 0.10 * (1.0 if postal_code_match > 0 or num_overlap_count > 0 else 0.0)
        )

    return {
        "name_fuzz_ratio": name_fuzz_ratio,
        "name_token_sort": name_token_sort,
        "name_token_set": name_token_set,
        "name_ngram_jaccard": name_ngram_jaccard,
        "name_word_jaccard": name_word_jaccard,
        "name_exact_match": name_exact_match,
        "name_len_diff": float(name_len_diff),
        "name_len_ratio": name_len_ratio,
        "addr_token_sort": addr_token_sort,
        "addr_token_set": addr_token_set,
        "addr_word_jaccard": addr_word_jaccard,
        "addr_num_jaccard": num_jaccard,
        "addr_num_overlap": num_overlap_count,
        "postal_code_match": postal_code_match,
        "is_addr_missing": is_addr_missing,
        "is_source2": is_source2,
        "composite_score": composite_score,
    }


def prepare_entity_profile(name: str, address: str) -> Dict[str, Any]:
    """
    Pre-computes and caches cleaned strings, n-grams, tokens, numbers, and postal codes
    for an entity record once, enabling ultra-fast sub-millisecond pairwise feature evaluation.
    """
    c_name = clean_business_name(name)
    c_addr = clean_address(address)
    name_ngrams = generate_char_ngrams(c_name, n=3)
    name_words = set(generate_word_tokens(c_name))
    addr_words = set(generate_word_tokens(c_addr))
    addr_nums = extract_numbers(c_addr)
    postal_codes = set(extract_postal_codes(c_addr))
    is_missing_addr = 1.0 if not c_addr or c_addr == "nan" else 0.0

    return {
        "clean_name": c_name,
        "clean_addr": c_addr,
        "name_ngrams": name_ngrams,
        "name_words": name_words,
        "addr_words": addr_words,
        "addr_nums": addr_nums,
        "postal_codes": postal_codes,
        "is_missing_addr": is_missing_addr,
        "name_len": len(c_name),
    }


def extract_pair_features_fast(
    p1: Dict[str, Any],
    p2: Dict[str, Any],
    source2_prefix: str = "S2",
) -> List[float]:
    """
    Ultra-fast vectorized similarity computation between two pre-computed entity profiles.
    """
    c_name1, c_name2 = p1["clean_name"], p2["clean_name"]
    c_addr1, c_addr2 = p1["clean_addr"], p2["clean_addr"]

    # Fast exact match shortcut
    if c_name1 and c_name1 == c_name2:
        name_fuzz_ratio = 1.0
        name_token_sort = 1.0
        name_token_set = 1.0
        name_ngram_jaccard = 1.0
        name_word_jaccard = 1.0
        name_exact_match = 1.0
        name_len_diff = 0.0
        name_len_ratio = 1.0
    else:
        name_exact_match = 0.0
        len1, len2 = p1["name_len"], p2["name_len"]
        name_len_diff = float(abs(len1 - len2))
        name_len_ratio = min(len1, len2) / max(len1, len2) if max(len1, len2) > 0 else 1.0

        name_token_set = fuzz.token_set_ratio(c_name1, c_name2) / 100.0
        if name_token_set < 0.35:
            # Short-circuit disjoint candidate pairs
            return [
                name_token_set,
                name_token_set,
                name_token_set,
                0.0,
                0.0,
                0.0,
                name_len_diff,
                name_len_ratio,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                p2["is_missing_addr"],
                1.0 if source2_prefix.startswith("S2") else 0.0,
            ]

        name_fuzz_ratio = fuzz.ratio(c_name1, c_name2) / 100.0
        name_token_sort = fuzz.token_sort_ratio(c_name1, c_name2) / 100.0
        name_ngram_jaccard = compute_jaccard_similarity(p1["name_ngrams"], p2["name_ngrams"])
        name_word_jaccard = compute_jaccard_similarity(p1["name_words"], p2["name_words"])

    is_addr_missing = p2["is_missing_addr"]
    if not is_addr_missing and c_addr1:
        addr_token_sort = fuzz.token_sort_ratio(c_addr1, c_addr2) / 100.0
        addr_token_set = fuzz.token_set_ratio(c_addr1, c_addr2) / 100.0
        addr_word_jaccard = compute_jaccard_similarity(p1["addr_words"], p2["addr_words"])

        nums1, nums2 = p1["addr_nums"], p2["addr_nums"]
        addr_num_jaccard = compute_jaccard_similarity(nums1, nums2)
        num_overlap_count = float(len(nums1.intersection(nums2)))

        pc1, pc2 = p1["postal_codes"], p2["postal_codes"]
        postal_code_match = 1.0 if (pc1 and pc2 and len(pc1.intersection(pc2)) > 0) else 0.0
    else:
        addr_token_sort = 0.0
        addr_token_set = 0.0
        addr_word_jaccard = 0.0
        addr_num_jaccard = 0.0
        num_overlap_count = 0.0
        postal_code_match = 0.0

    is_source2 = 1.0 if source2_prefix.startswith("S2") else 0.0

    return [
        name_fuzz_ratio,
        name_token_sort,
        name_token_set,
        name_ngram_jaccard,
        name_word_jaccard,
        name_exact_match,
        name_len_diff,
        name_len_ratio,
        addr_token_sort,
        addr_token_set,
        addr_word_jaccard,
        addr_num_jaccard,
        num_overlap_count,
        postal_code_match,
        is_addr_missing,
        is_source2,
    ]

