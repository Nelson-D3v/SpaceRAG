"""
vectorstore/chroma_store.py
───────────────────────────
Gerencia a criação, persistência e consulta do Vector Store ChromaDB.

Fluxo de indexação:
  1. Recebe lista de Documents (LangChain)
  2. Divide em chunks com RecursiveCharacterTextSplitter
  3. Gera embeddings de cada chunk
  4. Persiste no ChromaDB (diretório local)

Fluxo de consulta:
  1. Recebe query em texto
  2. Converte para embedding
  3. Busca os K chunks mais similares (cosine)
  4. Retorna como Retriever para o LangChain
"""

import logging
from pathlib import Path
from typing import List, Optional

import chromadb
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import (
    CHROMA_DIR,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    COLLECTION_NAME,
    RETRIEVAL_K,
)
from embeddings import get_embeddings

logger = logging.getLogger(__name__)


def split_documents(docs: List[Document]) -> List[Document]:
    """
    Divide documentos em chunks menores com sobreposição.

    A sobreposição (overlap) garante que trechos no limite entre
    dois chunks não percam contexto.

    Args:
        docs: Lista de Documents carregados.

    Returns:
        Lista de chunks (Documents menores).
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )
    chunks = splitter.split_documents(docs)
    logger.info(
        f"Chunking: {len(docs)} documento(s) → {len(chunks)} chunk(s) "
        f"(size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})"
    )
    return chunks


def get_or_create_vectorstore(
    documents: Optional[List[Document]] = None,
    persist_directory: Optional[str] = None,
) -> Chroma:
    """
    Retorna um Chroma Vector Store existente ou cria um novo.

    Se 'documents' for fornecido, os documentos são indexados.
    Se o diretório já contiver dados, o índice existente é carregado.

    Args:
        documents: Lista de Documents para indexar (opcional).
        persist_directory: Diretório de persistência (padrão: CHROMA_DIR).

    Returns:
        Instância de Chroma pronta para uso.
    """
    persist_dir = persist_directory or str(CHROMA_DIR)
    embeddings = get_embeddings()

    if documents:
        # Indexação: divide em chunks e persiste
        chunks = split_documents(documents)
        logger.info(f"Criando/atualizando Vector Store em: {persist_dir}")
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            collection_name=COLLECTION_NAME,
            persist_directory=persist_dir,
        )
        logger.info("Vector Store criado com sucesso.")
    else:
        # Carregamento: lê índice existente
        logger.info(f"Carregando Vector Store existente de: {persist_dir}")
        vectorstore = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=embeddings,
            persist_directory=persist_dir,
        )

    return vectorstore


def get_retriever(vectorstore: Chroma, k: int = RETRIEVAL_K):
    """
    Cria um Retriever a partir do Vector Store.

    O retriever é o componente que recebe a query em texto,
    calcula similaridade e devolve os K chunks mais relevantes.

    Args:
        vectorstore: Instância do Chroma Vector Store.
        k: Número de chunks a recuperar (padrão: RETRIEVAL_K).

    Returns:
        Retriever compatível com LangChain.
    """
    return vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k},
    )


def collection_exists(persist_directory: Optional[str] = None) -> bool:
    """
    Verifica se já existe uma coleção indexada no diretório especificado.

    Returns:
        True se a coleção existir e tiver documentos; False caso contrário.
    """
    persist_dir = persist_directory or str(CHROMA_DIR)
    try:
        client = chromadb.PersistentClient(path=persist_dir)
        collection = client.get_collection(COLLECTION_NAME)
        count = collection.count()
        logger.info(f"Coleção existente com {count} chunk(s).")
        return count > 0
    except Exception:
        return False


def get_collection_info(persist_directory: Optional[str] = None) -> dict:
    """
    Retorna informações sobre a coleção atual.

    Returns:
        Dict com nome, contagem de chunks e diretório.
    """
    persist_dir = persist_directory or str(CHROMA_DIR)
    try:
        client = chromadb.PersistentClient(path=persist_dir)
        collection = client.get_collection(COLLECTION_NAME)
        return {
            "collection": COLLECTION_NAME,
            "chunks": collection.count(),
            "directory": persist_dir,
        }
    except Exception:
        return {"collection": COLLECTION_NAME, "chunks": 0, "directory": persist_dir}
