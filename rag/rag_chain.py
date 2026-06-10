"""
rag/rag_chain.py
────────────────
Núcleo da arquitetura RAG.

Monta a RetrievalQA Chain do LangChain conectando:
  retriever (ChromaDB) → prompt template → LLM (Groq/Llama 3)

A chain funciona assim para cada pergunta:
  1. A query é convertida em embedding pelo retriever
  2. Os K chunks mais similares são recuperados do ChromaDB
  3. Os chunks são inseridos no prompt template como {context}
  4. O LLM gera a resposta usando o contexto recuperado
  5. A resposta retorna junto com os documentos fonte
"""

import logging
from typing import Dict, List, Optional

from langchain.chains import RetrievalQA
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq

from config import (
    GROQ_API_KEY,
    LLM_MAX_TOKENS,
    LLM_MODEL,
    LLM_TEMPERATURE,
    RAG_PROMPT_TEMPLATE,
    RETRIEVAL_K,
)

logger = logging.getLogger(__name__)


def build_llm() -> ChatGroq:
    """
    Instancia o LLM via API Groq (Llama 3 / Mixtral).

    A API Groq oferece acesso gratuito a modelos open-source de alto desempenho.
    Velocidade de inferência muito superior ao ChatGPT (tokens/s).

    Returns:
        Instância de ChatGroq configurada.

    Raises:
        ValueError: Se GROQ_API_KEY não estiver configurada.
    """
    if not GROQ_API_KEY:
        raise ValueError(
            "GROQ_API_KEY não configurada. "
            "Crie uma conta gratuita em https://console.groq.com e "
            "adicione a chave no arquivo .env"
        )

    logger.info(f"Inicializando LLM: {LLM_MODEL}")
    return ChatGroq(
        api_key=GROQ_API_KEY,
        model_name=LLM_MODEL,
        temperature=LLM_TEMPERATURE,
        max_tokens=LLM_MAX_TOKENS,
    )


def build_prompt() -> PromptTemplate:
    """
    Cria o PromptTemplate para a chain RAG.

    O template instrui o modelo a:
    - Usar apenas o contexto fornecido (grounding)
    - Responder em português
    - Admitir quando não souber a resposta (reduz alucinações)

    Returns:
        PromptTemplate com variáveis {context} e {question}.
    """
    return PromptTemplate(
        input_variables=["context", "question"],
        template=RAG_PROMPT_TEMPLATE,
    )


def build_rag_chain(retriever) -> RetrievalQA:
    """
    Monta a RetrievalQA Chain completa.

    Args:
        retriever: Retriever do LangChain (retorna chunks do ChromaDB).

    Returns:
        Chain pronta para receber perguntas via .invoke({"query": "..."}).
    """
    llm = build_llm()
    prompt = build_prompt()

    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",            # "stuff" = concatena todos os chunks no prompt
        retriever=retriever,
        return_source_documents=True,  # Retorna os chunks usados na resposta
        chain_type_kwargs={"prompt": prompt},
    )
    logger.info("RAG Chain montada com sucesso.")
    return chain


def ask(chain: RetrievalQA, question: str) -> Dict:
    """
    Executa uma pergunta na chain RAG e retorna resposta + fontes.

    Args:
        chain: RetrievalQA Chain montada por build_rag_chain().
        question: Pergunta do usuário em linguagem natural.

    Returns:
        Dict com:
          - "answer": texto da resposta gerada pelo LLM
          - "sources": lista de Documents recuperados (com metadata)
          - "question": pergunta original
    """
    logger.info(f"Processando pergunta: {question[:80]}...")

    result = chain.invoke({"query": question})

    answer = result.get("result", "Não foi possível gerar uma resposta.")
    source_docs: List[Document] = result.get("source_documents", [])

    # Formata informações das fontes
    sources = []
    seen = set()
    for doc in source_docs:
        source_name = doc.metadata.get("source", "desconhecido")
        page = doc.metadata.get("page", "—")
        key = f"{source_name}::{page}"
        if key not in seen:
            seen.add(key)
            sources.append({
                "source": source_name,
                "page": page,
                "preview": doc.page_content[:200] + "...",
            })

    logger.info(f"Resposta gerada. Fontes: {[s['source'] for s in sources]}")
    return {
        "answer": answer,
        "sources": sources,
        "question": question,
    }
