"""
GPU-Accelerated Dense Vector Retriever for Entity Resolution.
Supports Qwen3-Embedding / GTE-Qwen / BGE-M3 models with Matryoshka 512-dim truncation and FAISS-GPU cosine search.
"""

from typing import Any, Dict, List, Optional, Set, Tuple, Union
import numpy as np
import pandas as pd
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    TORCH_AVAILABLE = False

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

from ..preprocessing.text import clean_address, clean_business_name


class GPUDenseRetriever:
    """
    High-Throughput Dense Embedding & Similarity Search Engine.
    Encodes text into normalized dense vectors and queries nearest neighbors using FAISS.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-Embedding-0.6B",
        fallback_model_name: str = "BAAI/bge-m3",
        dim: int = 512,
        batch_size: int = 512,
        device: Optional[str] = None,
    ):
        if device is None:
            self.device = "cuda" if (TORCH_AVAILABLE and torch.cuda.is_available()) else "cpu"
        else:
            self.device = device

        self.model_name = model_name
        self.fallback_model_name = fallback_model_name
        self.dim = dim
        self.batch_size = batch_size
        self.model = None
        self.gpu_res = None

        if self.device == "cuda" and FAISS_AVAILABLE and hasattr(faiss, "StandardGpuResources"):
            try:
                self.gpu_res = faiss.StandardGpuResources()
            except Exception:
                self.gpu_res = None

    def _load_model(self):
        """Lazy loads the embedding model onto GPU/CPU in FP16."""
        if self.model is not None:
            return

        print(f"  [DenseRetriever] Initializing {self.model_name} on {self.device}...")
        try:
            self.model = SentenceTransformer(
                self.model_name,
                trust_remote_code=True,
                device=self.device,
                model_kwargs={"torch_dtype": torch.float16 if self.device == "cuda" else torch.float32},
            )
            print(f"  ✓ Loaded {self.model_name}")
        except Exception as e:
            print(f"  ⚠️ Warning: Failed to load {self.model_name} ({e}). Falling back to {self.fallback_model_name}...")
            self.model = SentenceTransformer(
                self.fallback_model_name,
                trust_remote_code=True,
                device=self.device,
                model_kwargs={"torch_dtype": torch.float16 if self.device == "cuda" else torch.float32},
            )
            print(f"  ✓ Loaded fallback {self.fallback_model_name}")

    def format_texts(self, df: pd.DataFrame, is_query: bool = False) -> List[str]:
        """
        Formats entity records into standardized strings with optional asymmetric query instructions.
        """
        names = df["business_name"].values
        addrs = df["business_address"].values
        countries = df["country"].values if "country" in df.columns else [""] * len(df)

        formatted = []
        for name, addr, country in zip(names, addrs, countries):
            c_name = clean_business_name(str(name))
            c_addr = clean_address(str(addr))
            c_country = str(country).strip()

            base_text = f"Business: {c_name} | Address: {c_addr} | Country: {c_country}"
            if is_query:
                # Asymmetric instruction prompt for Qwen3-Embedding queries
                prompt = f"Instruct: Retrieve matching business identity records across noisy sources\nQuery: {base_text}"
                formatted.append(prompt)
            else:
                formatted.append(base_text)

        return formatted

    def encode_texts(
        self,
        texts: List[str],
        is_query: bool = False,
        show_progress_bar: bool = False,
    ) -> np.ndarray:
        """
        Encodes texts into L2-normalized dense embeddings with Matryoshka dimension truncation.
        """
        self._load_model()
        
        # Batch encode with PyTorch
        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=show_progress_bar,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )

        # Matryoshka dimension truncation (e.g. 512 dims)
        if self.dim and embeddings.shape[1] > self.dim:
            embeddings = embeddings[:, :self.dim]
            # Re-normalize after truncation
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            embeddings = embeddings / norms

        return embeddings.astype(np.float32)

    def build_index(self, target_df: pd.DataFrame, show_progress_bar: bool = True):
        """
        Builds FAISS cosine index for target records and caches in memory/GPU.
        """
        self._load_model()
        self.target_eids = [str(x).strip() for x in target_df["entity_id"].values]
        target_texts = self.format_texts(target_df, is_query=False)
        target_embeddings = self.encode_texts(target_texts, is_query=False, show_progress_bar=show_progress_bar)
        self.current_index = self.build_faiss_index(target_embeddings)

    def query_candidates(
        self,
        query_df: pd.DataFrame,
        top_k: int = 6,
        show_progress_bar: bool = False,
    ) -> Dict[str, Set[str]]:
        """
        Queries the pre-built FAISS index using query embeddings.
        """
        if not hasattr(self, "current_index") or self.current_index is None:
            return {str(x).strip(): set() for x in query_df["entity_id"].values}

        query_eids = [str(x).strip() for x in query_df["entity_id"].values]
        query_texts = self.format_texts(query_df, is_query=True)
        q_embs = self.encode_texts(query_texts, is_query=True, show_progress_bar=show_progress_bar)

        _, top_indices = self.current_index.search(q_embs, top_k)

        candidates: Dict[str, Set[str]] = {}
        for q_id, top_idx_row in zip(query_eids, top_indices):
            cand_set = {self.target_eids[idx] for idx in top_idx_row if 0 <= idx < len(self.target_eids)}
            candidates[q_id] = cand_set

        return candidates

    def query_dense_candidates(
        self,
        query_df: pd.DataFrame,
        target_df: pd.DataFrame,
        top_k: int = 8,
    ) -> Dict[str, Set[str]]:
        """Legacy helper for backward compatibility."""
        self.build_index(target_df, show_progress_bar=True)
        return self.query_candidates(query_df, top_k=top_k, show_progress_bar=False)

