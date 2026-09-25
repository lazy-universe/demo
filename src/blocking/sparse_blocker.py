"""
Learned Sparse Retrieval Blocker using Linkup SparseUp / SPLADE.
Generates term-weight sparse representations to capture acronyms, synonyms, and rare lexical terms.
"""

from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np
import pandas as pd
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    TORCH_AVAILABLE = False

try:
    from transformers import AutoModelForMaskedLM, AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

from ..preprocessing.text import clean_address, clean_business_name


class LearnedSparseBlocker:
    """
    Learned Sparse Lexical Blocker.
    Expands entity texts into vocabulary-weighted sparse activations for inverted index lookup.
    """

    def __init__(
        self,
        model_name: str = "Linkup-Platform/linkup-sparseup-embed-v1",
        fallback_model_name: str = "naver/splade-v3",
        top_k_terms: int = 30,
        device: Optional[str] = None,
    ):
        if device is None:
            self.device = "cuda" if (TORCH_AVAILABLE and torch.cuda.is_available()) else "cpu"
        else:
            self.device = device

        self.model_name = model_name
        self.fallback_model_name = fallback_model_name
        self.top_k_terms = top_k_terms
        self.tokenizer = None
        self.model = None
        self.sparse_index: Dict[int, List[Tuple[str, float]]] = defaultdict(list)

    def _load_model(self):
        if self.model is not None or not TRANSFORMERS_AVAILABLE:
            return

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
            self.model = AutoModelForMaskedLM.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            ).to(self.device)
            self.model.eval()
        except Exception:
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(self.fallback_model_name, trust_remote_code=True)
                self.model = AutoModelForMaskedLM.from_pretrained(
                    self.fallback_model_name,
                    trust_remote_code=True,
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                ).to(self.device)
                self.model.eval()
            except Exception:
                self.model = None

    def extract_sparse_vectors(self, texts: List[str], batch_size: int = 128) -> List[Dict[int, float]]:
        """
        Computes sparse term weights using ReLU log activations.
        """
        self._load_model()
        if self.model is None:
            return []

        sparse_vectors = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            inputs = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors="pt"
            ).to(self.device)

            outputs = self.model(**inputs)
            logits = outputs.logits
            # SPLADE / SparseUp max-pooling over sequence
            values, _ = torch.max(torch.log(1 + torch.relu(logits)) * inputs.attention_mask.unsqueeze(-1), dim=1)
            
            for row in values:
                nonzero_indices = torch.nonzero(row).squeeze(-1)
                if len(nonzero_indices) > self.top_k_terms:
                    top_vals, top_idx = torch.topk(row[nonzero_indices], self.top_k_terms)
                    term_dict = {int(idx): float(val) for idx, val in zip(nonzero_indices[top_idx].cpu(), top_vals.cpu())}
                else:
                    term_dict = {int(idx): float(row[idx].item()) for idx in nonzero_indices.cpu()}
                sparse_vectors.append(term_dict)

        return sparse_vectors

    def build_index(self, target_df: pd.DataFrame):
        """Indexes target records in a sparse inverted index."""
        self.sparse_index.clear()
        texts = [f"{clean_business_name(str(n))} {clean_address(str(a))}" for n, a in zip(target_df["business_name"], target_df["business_address"])]
        eids = [str(eid).strip() for eid in target_df["entity_id"]]

        sparse_vecs = self.extract_sparse_vectors(texts)
        for eid, svec in zip(eids, sparse_vecs):
            for token_id, weight in svec.items():
                self.sparse_index[token_id].append((eid, weight))

    def query_candidates(self, query_df: pd.DataFrame, top_k: int = 4) -> Dict[str, Set[str]]:
        """Queries sparse index and returns top_k candidates by dot-product score."""
        texts = [f"{clean_business_name(str(n))} {clean_address(str(a))}" for n, a in zip(query_df["business_name"], query_df["business_address"])]
        query_eids = [str(eid).strip() for eid in query_df["entity_id"]]

        sparse_vecs = self.extract_sparse_vectors(texts)
        candidates: Dict[str, Set[str]] = {}

        for q_id, svec in zip(query_eids, sparse_vecs):
            scores = defaultdict(float)
            for token_id, q_weight in svec.items():
                if token_id in self.sparse_index:
                    for target_id, t_weight in self.sparse_index[token_id]:
                        scores[target_id] += q_weight * t_weight

            if scores:
                sorted_cands = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)[:top_k]
                candidates[q_id] = set(sorted_cands)
            else:
                candidates[q_id] = set()

        return candidates
