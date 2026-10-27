"""Load and invoke BGE-M3 dense embeddings (1024 dimensions) from the local model cache.

Weights are never downloaded at runtime; fetch them once with `make embedding-model`.
Inference runs in a worker thread so the event loop keeps heartbeating leases.
"""
import asyncio
import os
from pathlib import Path
import threading

CACHE = Path(__file__).resolve().parents[2] / '.cache' / 'huggingface'


class EmbeddingUnavailable(Exception):
    pass


class BGEM3:
    def __init__(self, settings):
        self.settings = settings
        self._tokenizer = self._model = self._device = None
        self._lock = threading.Lock()
        # transformers otherwise starts a background Hub lookup to convert .bin weights to
        # safetensors, even with local_files_only; the worker must not call out at runtime.
        os.environ.setdefault('DISABLE_SAFETENSORS_CONVERSION', '1')
        os.environ.setdefault('HF_HUB_DISABLE_TELEMETRY', '1')

    def _load_tokenizer(self):
        if self._tokenizer is None:
            from transformers import AutoTokenizer
            try:
                self._tokenizer = AutoTokenizer.from_pretrained(
                    self.settings.embedding_model, revision=self.settings.embedding_model_revision,
                    cache_dir=CACHE / 'hub', local_files_only=True)
            except OSError:
                raise EmbeddingUnavailable('Embedding model is not in the local cache; run make embedding-model.') from None
        return self._tokenizer

    def _load_model(self):
        if self._model is None:
            import torch
            from transformers import AutoModel
            device = self.settings.embedding_device
            if device == 'auto':
                device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
            try:
                model = AutoModel.from_pretrained(
                    self.settings.embedding_model, revision=self.settings.embedding_model_revision,
                    cache_dir=CACHE / 'hub', local_files_only=True)
            except OSError:
                raise EmbeddingUnavailable('Embedding model is not in the local cache; run make embedding-model.') from None
            self._model, self._device = model.to(device).eval(), device
        return self._model

    def count_tokens(self, text: str) -> int:
        return len(self._load_tokenizer()(text, add_special_tokens=False)['input_ids'])

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Dense BGE-M3 vectors: [CLS] pooling, L2-normalised, as the model card specifies."""
        import torch
        with self._lock:
            tokenizer, model = self._load_tokenizer(), self._load_model()
            batch = tokenizer(texts, padding=True, truncation=True, max_length=self.settings.embedding_max_length,
                              return_tensors='pt').to(self._device)
            with torch.inference_mode():
                vectors = model(**batch).last_hidden_state[:, 0]
                vectors = torch.nn.functional.normalize(vectors, p=2, dim=-1)
            return vectors.float().cpu().tolist()

    async def embed_async(self, texts: list[str]) -> list[list[float]]:
        return await asyncio.to_thread(self.embed, texts)
