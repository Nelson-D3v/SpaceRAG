"""
config.py
─────────
Configurações centralizadas da aplicação.
Lê variáveis do arquivo .env e expõe constantes para todos os módulos.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Carrega .env da raiz do projeto
load_dotenv()

# ── Caminhos ──────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
DOCS_DIR = BASE_DIR / "docs"
CHROMA_DIR = Path(os.getenv("CHROMA_PERSIST_DIR", "./vectorstore/chroma_db"))

# Garante que os diretórios existam
DOCS_DIR.mkdir(parents=True, exist_ok=True)
CHROMA_DIR.mkdir(parents=True, exist_ok=True)

# ── LLM ───────────────────────────────────────────────────────
GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
LLM_MODEL: str = os.getenv("LLM_MODEL", "llama-3.1-70b-versatile")
LLM_TEMPERATURE: float = 0.2          # Baixo para respostas factuais
LLM_MAX_TOKENS: int = 1024

# ── Embeddings ────────────────────────────────────────────────
EMBEDDING_MODEL: str = os.getenv(
    "EMBEDDING_MODEL",
    "sentence-transformers/all-MiniLM-L6-v2"
)

# ── Chunking ──────────────────────────────────────────────────
CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", 1000))
CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", 200))

# ── Retriever ─────────────────────────────────────────────────
RETRIEVAL_K: int = int(os.getenv("RETRIEVAL_K", 4))

# ── Coleção ChromaDB ──────────────────────────────────────────
COLLECTION_NAME: str = "space_economy_docs"

# ── Extensões suportadas ──────────────────────────────────────
SUPPORTED_EXTENSIONS: list[str] = [".pdf", ".txt", ".docx"]

# ── Template do Prompt RAG ────────────────────────────────────
RAG_PROMPT_TEMPLATE: str = """
Você é um assistente especializado em Nova Economia Espacial.
Use APENAS as informações do contexto abaixo para responder à pergunta.
Se a resposta não estiver no contexto, diga "Não encontrei essa informação nos documentos carregados."
Responda sempre em português do Brasil, de forma clara e objetiva.

CONTEXTO:
{context}

PERGUNTA:
{question}

RESPOSTA:"""
