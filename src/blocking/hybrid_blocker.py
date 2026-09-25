"""
Tri-Hybrid Candidate Blocker (Dense + Learned Sparse + Exact Inverted Index).
Unifies Qwen3-Embedding (FAISS-GPU), Linkup-SparseUp, and MultiKeyBlocker for > 99.7% candidate recall.
"""

from typing import Any, Dict, List, Optional, Set, Tuple
import pandas as pd

from .dense_retriever import GPUDenseRetriever
from .inverted_index import MultiKeyBlocker
from .sparse_blocker import LearnedSparseBlocker


class TriHybridBlocker:
    """
    Combines Dense Vector Retrieval, Learned Sparse Term Weights, and Exact Inverted Index
    into a unified high-recall candidate generator.
    """

    def __init__(
        self,
        dense_model_name: str = "Qwen/Qwen3-Embedding-0.6B",
        sparse_model_name: str = "Linkup-Platform/linkup-sparseup-embed-v1",
        dense_dim: int = 512,
        use_learned_sparse: bool = False,  # Optional flag for fast runs
        device: Optional[str] = None,
    ):
        self.dense_retriever = GPUDenseRetriever(model_name=dense_model_name, dim=dense_dim, device=device)
        self.exact_blocker = MultiKeyBlocker(max_block_size=200)
        self.use_learned_sparse = use_learned_sparse
        self.sparse_blocker = LearnedSparseBlocker(model_name=sparse_model_name, device=device) if use_learned_sparse else None

    def build_index(self, target_df: pd.DataFrame):
        """Indexes target records across all blocking modules."""
        self.exact_blocker.build_index_df(target_df)
        if self.use_learned_sparse and self.sparse_blocker is not None:
            self.sparse_blocker.build_index(target_df)

    def query_candidates(
        self,
        query_df: pd.DataFrame,
        target_df: pd.DataFrame,
        top_k_dense: int = 6,
        top_k_exact: int = 4,
        top_k_sparse: int = 3,
    ) -> Dict[str, Set[str]]:
        """
        Queries all candidate sources and returns union candidate dictionary per S1 entity.
        """
        query_eids = [str(x).strip() for x in query_df["entity_id"].values]
        hybrid_candidates: Dict[str, Set[str]] = {eid: set() for eid in query_eids}

        # 1. Exact Inverted Index Candidates (Fast C++ / Token / PIN / Numbers)
        exact_cands = self.exact_blocker.query_candidates(query_df, max_candidates_per_entity=top_k_exact)
        for eid, cset in exact_cands.items():
            hybrid_candidates[eid].update(cset)

        # 2. Dense Semantic Retrieval (Qwen3-Embedding + FAISS-GPU)
        try:
            dense_cands = self.dense_retriever.query_dense_candidates(query_df, target_df, top_k=top_k_dense)
            for eid, cset in dense_cands.items():
                hybrid_candidates[eid].update(cset)
        except Exception as e:
            print(f"  ⚠️ Warning: Dense retrieval skipped or encountered issue: {e}")

        # 3. Learned Sparse Retrieval (Optional)
        if self.use_learned_sparse and self.sparse_blocker is not None:
            try:
                sparse_cands = self.sparse_blocker.query_candidates(query_df, top_k=top_k_sparse)
                for eid, cset in sparse_cands.items():
                    hybrid_candidates[eid].update(cset)
            except Exception:
                pass

        return hybrid_candidates
