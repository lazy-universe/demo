"""
Cross-Encoder Transformer Pairwise Reranker for Entity Resolution.
Supports Qwen3-Reranker-0.6B / GTE-Reranker / BGE-Reranker-v2-M3 (Apache 2.0).
Computes full sequence cross-attention on GPU in FP16 to generate calibrated match probabilities.
"""

from typing import Any, Dict, List, Optional, Set, Tuple, Union
import numpy as np
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    TORCH_AVAILABLE = False

try:
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

from ..preprocessing.text import clean_address, clean_business_name


class QwenTransformerReranker:
    """
    High-Precision Cross-Encoder Pairwise Reranker.
    Compares sequence pairs using full multi-head cross-attention.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-Reranker-0.6B",
        fallback_model_name: str = "BAAI/bge-reranker-v2-m3",
        max_length: int = 256,
        batch_size: int = 256,
        device: Optional[str] = None,
    ):
        if device is None:
            self.device = "cuda" if (TORCH_AVAILABLE and torch.cuda.is_available()) else "cpu"
        else:
            self.device = device

        self.model_name = model_name
        self.fallback_model_name = fallback_model_name
        self.max_length = max_length
        self.batch_size = batch_size
        self.tokenizer = None
        self.model = None

    def _load_model(self):
        if self.model is not None or not TRANSFORMERS_AVAILABLE:
            return

        print(f"  [Reranker] Loading cross-encoder {self.model_name} on {self.device}...")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            ).to(self.device)
            self.model.eval()
            print(f"  ✓ Loaded {self.model_name}")
        except Exception as e:
            print(f"  ⚠️ Warning: Failed to load {self.model_name} ({e}). Falling back to {self.fallback_model_name}...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.fallback_model_name, trust_remote_code=True)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.fallback_model_name,
                trust_remote_code=True,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            ).to(self.device)
            self.model.eval()
            print(f"  ✓ Loaded fallback {self.fallback_model_name}")

    def format_pair_text(self, name1: str, addr1: str, name2: str, addr2: str) -> Tuple[str, str]:
        """Formats record pair into standardized cross-encoder text inputs."""
        s1_text = f"Business: {clean_business_name(str(name1))} | Address: {clean_address(str(addr1))}"
        s2_text = f"Business: {clean_business_name(str(name2))} | Address: {clean_address(str(addr2))}"
        return s1_text, s2_text

    def predict_pair_probabilities(
        self,
        pairs: List[Tuple[str, str]],
        batch_size: Optional[int] = None,
        show_progress: bool = True,
    ) -> np.ndarray:
        """
        Predicts binary match probabilities P(match | pair) for a list of (text1, text2) tuples.
        Includes live tqdm progress bar showing throughput and ETA.
        """
        if not pairs:
            return np.array([], dtype=np.float32)

        self._load_model()
        if self.model is None:
            # Fallback simple scoring if model cannot be loaded
            return np.ones(len(pairs), dtype=np.float32) * 0.5

        bs = batch_size or self.batch_size
        probabilities = []

        try:
            from tqdm import tqdm
            iterator = range(0, len(pairs), bs)
            if show_progress and len(pairs) > bs:
                iterator = tqdm(iterator, desc="  ⚡ Qwen3-Reranker Inference", unit="batch", leave=False)
        except ImportError:
            iterator = range(0, len(pairs), bs)

        inference_ctx = torch.inference_mode if hasattr(torch, "inference_mode") else torch.no_grad

        with inference_ctx():
            for i in iterator:
                batch = pairs[i:i + bs]
                texts_a = [p[0] for p in batch]
                texts_b = [p[1] for p in batch]

                inputs = self.tokenizer(
                    texts_a,
                    texts_b,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                ).to(self.device)

                outputs = self.model(**inputs)
                logits = outputs.logits
                if logits.shape[-1] == 1:
                    probs = torch.sigmoid(logits.squeeze(-1)).detach().cpu().numpy()
                elif logits.shape[-1] == 2:
                    probs = torch.softmax(logits, dim=-1)[:, 1].detach().cpu().numpy()
                else:
                    probs = torch.sigmoid(logits[:, 0]).detach().cpu().numpy()

                if probs.ndim == 0:
                    probabilities.append(float(probs))
                else:
                    probabilities.extend(probs.tolist())

        return np.array(probabilities, dtype=np.float32)

