"""
Multi-Key Inverted Index Candidate Generation (Blocking) Engine.
Partitions dynamically by country and builds high-recall candidate buckets.
"""

from collections import defaultdict
from typing import Dict, List, Optional, Set
import pandas as pd

from ..config import MAX_BLOCK_SIZE
from ..preprocessing.text import clean_address, clean_business_name, clean_text
from ..preprocessing.tokenizer import extract_numbers, extract_postal_codes


class MultiKeyBlocker:
    """
    Multi-Key Inverted Index Blocker for Entity Resolution.
    Partitions strictly by country, and indexes records on:
    - Standardized Name Prefix (first 3-4 chars)
    - Distinctive Name Tokens (excluding generic stopwords)
    - Postal / PIN Codes
    - Shared Numerical Address Tokens
    """

    def __init__(
        self,
        max_block_size: int = MAX_BLOCK_SIZE,
        stop_words: Optional[Set[str]] = None,
    ):
        self.max_block_size = max_block_size
        self.stop_words = stop_words or {
            "the", "and", "co", "company", "services", "shop", "store", "center",
            "hotel", "restaurant", "solutions", "traders", "trading", "mart", "bazaar",
            "pvt", "ltd", "inc", "corp", "llc", "enterprises", "group", "associates",
        }
        self.index: Dict[str, List[str]] = defaultdict(list)

    def extract_keys(self, name: str, address: str, country: str) -> Set[str]:
        """
        Extracts distinctive blocking keys dynamically partitioned by country.
        """
        keys = set()
        c_clean = clean_text(country)
        name_clean = clean_business_name(name)
        addr_clean = clean_address(address)

        # 1. Name prefix key (first 4 characters)
        if name_clean:
            compact_name = name_clean.replace(" ", "")
            if len(compact_name) >= 3:
                prefix = compact_name[:4]
                keys.add(f"{c_clean}:np:{prefix}")

        # 2. Distinctive name word tokens
        words = name_clean.split()
        for w in words:
            if len(w) >= 3 and w not in self.stop_words:
                keys.add(f"{c_clean}:nw:{w}")

        # 3. Postal / PIN codes
        postal_codes = extract_postal_codes(address)
        for pc in postal_codes:
            keys.add(f"{c_clean}:pc:{pc}")

        # 4. Address Numbers combined with Name First Letter
        addr_nums = extract_numbers(address)
        first_char = name_clean[:1] if name_clean else "x"
        for num in addr_nums:
            if len(num) >= 2:
                keys.add(f"{c_clean}:num:{first_char}_{num}")

        return keys

    def build_index(self, s2_records: pd.DataFrame, s3_records: pd.DataFrame):
        """
        Populates the inverted index with candidate records from Source 2 and Source 3.
        Optimized for high-throughput sub-second indexing.
        """
        self.index.clear()

        for df in [s2_records, s3_records]:
            eids = df["entity_id"].values
            names = df["business_name"].values
            addrs = df["business_address"].values
            countries = df["country"].values

            for eid, name, addr, country in zip(eids, names, addrs, countries):
                eid_str = str(eid).strip()
                keys = self.extract_keys(str(name), str(addr), str(country))
                for k in keys:
                    self.index[k].append(eid_str)

        # Filter out oversized blocks (too uninformative / combinatorial explosion)
        if self.max_block_size > 0:
            for k in list(self.index.keys()):
                if len(self.index[k]) > self.max_block_size:
                    del self.index[k]

    def build_index_df(self, df: pd.DataFrame):
        """
        Populates the inverted index with candidate records from a single DataFrame slice.
        """
        self.index.clear()
        eids = df["entity_id"].values
        names = df["business_name"].values
        addrs = df["business_address"].values
        countries = df["country"].values

        for eid, name, addr, country in zip(eids, names, addrs, countries):
            eid_str = str(eid).strip()
            keys = self.extract_keys(str(name), str(addr), str(country))
            for k in keys:
                self.index[k].append(eid_str)

        if self.max_block_size > 0:
            for k in list(self.index.keys()):
                if len(self.index[k]) > self.max_block_size:
                    del self.index[k]

    # Alias for flexibility
    index_sources = build_index

    def query_candidates(
        self,
        s1_records: pd.DataFrame,
        max_candidates_per_entity: int = 35,
    ) -> Dict[str, Set[str]]:
        """
        Queries the inverted index for all Source 1 records to retrieve candidate matches.
        Optimized with fast array iteration.
        """
        candidates: Dict[str, Set[str]] = {}

        eids = s1_records["entity_id"].values
        names = s1_records["business_name"].values
        addrs = s1_records["business_address"].values
        countries = s1_records["country"].values

        for eid, name, addr, country in zip(eids, names, addrs, countries):
            s1_id = str(eid).strip()
            keys = self.extract_keys(str(name), str(addr), str(country))
            matched_pool = defaultdict(int)

            for k in keys:
                if k in self.index:
                    for target_id in self.index[k]:
                        matched_pool[target_id] += 1

            if not matched_pool:
                candidates[s1_id] = set()
            else:
                # Rank by number of shared blocking keys
                sorted_candidates = sorted(
                    matched_pool.keys(),
                    key=lambda x: matched_pool[x],
                    reverse=True,
                )
                candidates[s1_id] = set(sorted_candidates[:max_candidates_per_entity])

        return candidates

