"""Turns text into a vector so search can match by meaning, not only by spelling."""
from __future__ import annotations

import hashlib
import math
from functools import lru_cache

from . import config


class FastEmbedEmbedder:
    """Small multilingual sentence model, run through ONNX (no PyTorch, no GPU)."""

    def __init__(self) -> None:
        from fastembed import TextEmbedding

        self.model = TextEmbedding(
            model_name=config.EMBED_MODEL,
            cache_dir=str(config.MODELS_DIR / "fastembed"),
        )

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [_unit([float(x) for x in v]) for v in self.model.embed(texts)]


class HashEmbedder:
    """Offline stand-in used by the tests. It only captures spelling overlap,
    not meaning, so never use it for real search."""

    def embed(self, texts: list[str]) -> list[list[float]]:
        out = []
        for text in texts:
            v = [0.0] * config.EMBED_DIM
            t = f"  {text.lower()}  "
            for i in range(len(t) - 2):
                h = int.from_bytes(hashlib.md5(t[i:i + 3].encode()).digest()[:4], "big")
                v[h % config.EMBED_DIM] += 1.0
            out.append(_unit(v))
        return out


def _unit(v: list[float]) -> list[float]:
    norm = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / norm for x in v]


@lru_cache(maxsize=1)
def get_embedder():
    if config.EMBED_BACKEND == "hash":
        return HashEmbedder()
    return FastEmbedEmbedder()


def embed_one(text: str) -> list[float]:
    return get_embedder().embed([text or " "])[0]
