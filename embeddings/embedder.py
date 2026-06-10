"""
embeddings/embedder.py
──────────────────────
Gerencia o modelo de embeddings utilizado pela aplicação.

Utiliza HuggingFace Sentence Transformers, que roda LOCALMENTE
(sem custos de API). O modelo padrão é all-MiniLM-L6-v2:
  • Dimensão: 384
  • Multilingual: bom desempenho em português
  • Tamanho: ~90 MB (download único)

O objeto HuggingFaceEmbeddings é compatível com a interface
Embeddings do LangChain — pode ser trocado por OpenAIEmbeddings
sem alterar o restante do código.
"""

import logging
from functools import lru_cache
from typing import Optional

from langchain_huggingface import HuggingFaceEmbeddings

from config import EMBEDDING_MODEL

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_embeddings(model_name: Optional[str] = None) -> HuggingFaceEmbeddings:
    """
    Retorna uma instância (cacheada) do modelo de embeddings.

    O decorator @lru_cache garante que o modelo é carregado apenas
    uma vez por processo — evita recarregamentos a cada requisição.

    Args:
        model_name: Nome do modelo HuggingFace (opcional; usa EMBEDDING_MODEL do .env).

    Returns:
        Instância de HuggingFaceEmbeddings pronta para uso.
    """
    name = model_name or EMBEDDING_MODEL
    logger.info(f"Carregando modelo de embeddings: {name}")

    embeddings = HuggingFaceEmbeddings(
        model_name=name,
        model_kwargs={"device": "cpu"},   # Troque por "cuda" se tiver GPU
        encode_kwargs={
            "normalize_embeddings": True,  # Normaliza para cosine similarity
            "batch_size": 32,
        },
    )
    logger.info("Modelo de embeddings pronto.")
    return embeddings
