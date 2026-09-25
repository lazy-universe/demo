"""
Cross-Encoder Transformer Pairwise Reranker for Entity Resolution.
Supports BAAI/bge-reranker-v2-m3 / Qwen3-Reranker (Apache 2.0).
Computes full sequence cross-attention on GPU in FP16 to generate calibrated match probabilities.
"""

from typing import Any, Dict, List, Optional, Set, Tuple, Union
import os
import time
import numpy as np

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    TORCH_AVAILABLE = False

try:
    from sentence_transformers import CrossEncoder
    CROSS_ENCODER_AVAILABLE = True
except ImportError:
    CROSS_ENCODER_AVAILABLE = False

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
        model_name: str = "BAAI/bge-reranker-v2-m3",
        fallback_model_name: str = "BAAI/bge-reranker-v2-m3",
        max_length: int = 160,
        batch_size: int = 512,
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
        self.cross_encoder = None
        self.tokenizer = None
        self.model = None

    def _load_model(self):
        """Eagerly loads and caches the cross-encoder model onto GPU/CPU in FP16."""
        if self.cross_encoder is not None or self.model is not None:
            return

        t0 = time.time()
        print(f"  [Reranker] Loading Cross-Encoder {self.model_name} on {self.device}...")

        # 1. Try SentenceTransformers CrossEncoder (Preferred & highly optimized)
        if CROSS_ENCODER_AVAILABLE:
            try:
                model_kwargs = {"torch_dtype": torch.float16 if self.device == "cuda" else torch.float32}
                self.cross_encoder = CrossEncoder(
                    self.model_name,
                    max_length=self.max_length,
                    device=self.device,
                    automodel_args=model_kwargs,
                    trust_remote_code=True,
                )
                print(f"  ✓ Loaded {self.model_name} via SentenceTransformers CrossEncoder in {time.time()-t0:.2f}s")
                return
            except Exception as e:
                print(f"  ℹ️ CrossEncoder notice: {e}. Trying transformers AutoModel...")

        # 2. Fallback to Transformers AutoModelForSequenceClassification
        if TRANSFORMERS_AVAILABLE:
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token or "<|endoftext|>"
                    self.tokenizer.pad_token_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.pad_token)

                self.model = AutoModelForSequenceClassification.from_pretrained(
                    self.model_name,
                    trust_remote_code=True,
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                )
                if hasattr(self.model, "config") and getattr(self.model.config, "pad_token_id", None) is None:
                    self.model.config.pad_token_id = self.tokenizer.pad_token_id

                self.model.to(self.device)
                self.model.eval()
                print(f"  ✓ Loaded {self.model_name} via Transformers in {time.time()-t0:.2f}s")
                return
            except Exception as e:
                print(f"  ⚠️ Warning: Failed to load {self.model_name} ({e}). Falling back to {self.fallback_model_name}...")
                self.tokenizer = AutoTokenizer.from_pretrained(self.fallback_model_name, trust_remote_code=True)
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token or "<|endoftext|>"
                    self.tokenizer.pad_token_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.pad_token)

                self.model = AutoModelForSequenceClassification.from_pretrained(
                    self.fallback_model_name,
                    trust_remote_code=True,
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                )
                if hasattr(self.model, "config") and getattr(self.model.config, "pad_token_id", None) is None:
                    self.model.config.pad_token_id = self.tokenizer.pad_token_id

                self.model.to(self.device)
                self.model.eval()
                print(f"  ✓ Loaded fallback {self.fallback_model_name} in {time.time()-t0:.2f}s")

    def warmup(self):
        """Warm up GPU memory and verify inference execution."""
        self._load_model()
        test_pairs = [
            ("Business: Amazon HQ | Address: Seattle WA USA", "Business: Amazon Web Services | Address: Seattle Washington USA"),
            ("Business: Cafe Bakery | Address: Paris France", "Business: Steel Manufacturing | Address: Mumbai India"),
        ]
        test_probs = self.predict_pair_probabilities(test_pairs, show_progress=False)
        print(f"  ✓ Reranker GPU warm-up successful (Sample scores: {test_probs[0]:.4f} vs {test_probs[1]:.4f})")

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
        bs = batch_size or self.batch_size

        # 1. Native CrossEncoder inference
        if self.cross_encoder is not None:
            raw_scores = self.cross_encoder.predict(
                pairs,
                batch_size=bs,
                show_progress_bar=show_progress,
                convert_to_numpy=True,
            )
            # Sigmoid conversion if raw logits
            if np.min(raw_scores) < 0.0 or np.max(raw_scores) > 1.0:
                probs = 1.0 / (1.0 + np.exp(-raw_scores))
            else:
                probs = raw_scores
            return probs.astype(np.float32)

        # 2. Transformers SequenceClassification inference
        if self.model is None:
            return np.ones(len(pairs), dtype=np.float32) * 0.5

        probabilities = []
        try:
            from tqdm import tqdm
            iterator = range(0, len(pairs), bs)
            if show_progress and len(pairs) > bs:
                iterator = tqdm(iterator, desc="  ⚡ Reranker Inference", unit="batch", leave=False)
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
