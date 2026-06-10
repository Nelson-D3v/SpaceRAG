"""
app.py
──────
Interface principal da aplicação — Assistente RAG para Nova Economia Espacial.

Execução:
    streamlit run app.py

A interface é dividida em três áreas:
  • Sidebar  — upload de documentos + status do sistema
  • Main     — chat interativo com histórico
  • Expander — fontes consultadas para cada resposta
"""

import logging
import sys
import time
from pathlib import Path

import streamlit as st

# ── Configuração de logging ───────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# ── Importações internas ──────────────────────────────────────
# sys.path garante que os módulos locais sejam encontrados
sys.path.insert(0, str(Path(__file__).parent))

from loaders import load_document
from vectorstore import (
    get_or_create_vectorstore,
    get_retriever,
    collection_exists,
    get_collection_info,
)
from rag import build_rag_chain, ask
from utils import (
    save_uploaded_file,
    validate_file_extension,
    format_sources_markdown,
    cleanup_temp_file,
    get_file_icon,
)

# ── Configuração da página Streamlit ─────────────────────────
st.set_page_config(
    page_title="SpaceRAG — Assistente Espacial",
    page_icon="🛸",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS customizado ───────────────────────────────────────────
st.markdown("""
<style>
/* Cabeçalho principal */
.main-header {
    background: linear-gradient(135deg, #0d1b2a 0%, #1b2d4f 50%, #0d1b2a 100%);
    padding: 1.5rem 2rem;
    border-radius: 12px;
    margin-bottom: 1.5rem;
    border: 1px solid #1e3a5f;
}
.main-header h1 { color: #e8f4f8; margin: 0; font-size: 1.8rem; }
.main-header p  { color: #8ab4cc; margin: 0.25rem 0 0; font-size: 0.95rem; }

/* Balões de chat */
.chat-user {
    background: #1b2d4f;
    border-left: 3px solid #4a9eff;
    padding: 0.75rem 1rem;
    border-radius: 0 8px 8px 0;
    margin: 0.5rem 0;
}
.chat-assistant {
    background: #12271a;
    border-left: 3px solid #2ecc71;
    padding: 0.75rem 1rem;
    border-radius: 0 8px 8px 0;
    margin: 0.5rem 0;
}

/* Badge de status */
.status-ok   { color: #2ecc71; font-weight: bold; }
.status-warn { color: #f39c12; font-weight: bold; }
.status-err  { color: #e74c3c; font-weight: bold; }

/* Métricas na sidebar */
.metric-card {
    background: #1b2d4f;
    border-radius: 8px;
    padding: 0.6rem 1rem;
    margin: 0.3rem 0;
    font-size: 0.85rem;
    color: #cce5ff;
}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# ESTADO DA SESSÃO
# ══════════════════════════════════════════════════════════════
def init_session_state():
    """Inicializa variáveis de estado da sessão Streamlit."""
    defaults = {
        "chat_history": [],       # Lista de dicts {role, content, sources}
        "rag_chain": None,        # Chain RAG montada
        "vectorstore": None,      # Instância ChromaDB
        "docs_processed": [],     # Nomes dos documentos já processados
        "processing": False,      # Flag de processamento em andamento
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


init_session_state()


# ══════════════════════════════════════════════════════════════
# FUNÇÕES DE PROCESSAMENTO
# ══════════════════════════════════════════════════════════════
def process_documents(uploaded_files) -> bool:
    """
    Processa os documentos enviados pelo usuário:
    1. Salva temporariamente no disco
    2. Carrega com o loader adequado
    3. Indexa no ChromaDB
    4. Monta a RAG Chain

    Returns:
        True se processamento bem-sucedido, False caso contrário.
    """
    all_docs = []
    temp_files = []

    progress = st.progress(0, text="Iniciando processamento...")

    try:
        # Etapa 1: Carregar documentos
        for i, uploaded_file in enumerate(uploaded_files):
            progress.progress(
                int((i / len(uploaded_files)) * 40),
                text=f"Carregando {uploaded_file.name}..."
            )

            if not validate_file_extension(uploaded_file.name):
                st.warning(f"⚠️ {uploaded_file.name}: formato não suportado. Ignorado.")
                continue

            tmp_path = save_uploaded_file(uploaded_file)
            temp_files.append(tmp_path)

            docs = load_document(tmp_path)
            # Preserva o nome original no metadata
            for doc in docs:
                doc.metadata["source"] = uploaded_file.name
            all_docs.extend(docs)

        if not all_docs:
            st.error("Nenhum documento válido foi carregado.")
            return False

        # Etapa 2: Indexar no ChromaDB
        progress.progress(50, text="Gerando embeddings e indexando...")
        vectorstore = get_or_create_vectorstore(documents=all_docs)
        st.session_state.vectorstore = vectorstore

        # Etapa 3: Criar retriever
        progress.progress(75, text="Configurando retriever...")
        retriever = get_retriever(vectorstore)

        # Etapa 4: Montar RAG Chain
        progress.progress(90, text="Inicializando LLM e montando chain RAG...")
        chain = build_rag_chain(retriever)
        st.session_state.rag_chain = chain

        # Registra documentos processados
        st.session_state.docs_processed.extend(
            [f.name for f in uploaded_files]
        )

        progress.progress(100, text="✅ Processamento concluído!")
        time.sleep(0.5)
        progress.empty()
        return True

    except Exception as e:
        logger.error(f"Erro no processamento: {e}", exc_info=True)
        st.error(f"❌ Erro ao processar documentos: {str(e)}")
        progress.empty()
        return False

    finally:
        # Remove arquivos temporários
        for tmp_path in temp_files:
            cleanup_temp_file(tmp_path)


# ══════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🛸 SpaceRAG")
    st.markdown("*Assistente para Nova Economia Espacial*")
    st.divider()

    # ── Upload de documentos ─────────────────────────────────
    st.markdown("### 📁 Carregar Documentos")
    st.caption("Formatos: PDF, TXT, DOCX")

    uploaded_files = st.file_uploader(
        label="Selecione os documentos",
        type=["pdf", "txt", "docx"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    if uploaded_files:
        st.markdown(f"**{len(uploaded_files)} arquivo(s) selecionado(s):**")
        for f in uploaded_files:
            icon = get_file_icon(f.name)
            size_kb = len(f.getvalue()) / 1024
            st.markdown(f"{icon} `{f.name}` ({size_kb:.1f} KB)")

        btn_process = st.button(
            "⚡ Processar Documentos",
            use_container_width=True,
            type="primary",
        )

        if btn_process:
            with st.spinner("Processando..."):
                success = process_documents(uploaded_files)
                if success:
                    info = get_collection_info()
                    st.success(
                        f"✅ {len(uploaded_files)} documento(s) indexado(s)!\n\n"
                        f"Total de chunks: **{info['chunks']}**"
                    )
    st.divider()

    # ── Status do Sistema ────────────────────────────────────
    st.markdown("### 📊 Status do Sistema")

    chain_status = "✅ Pronto" if st.session_state.rag_chain else "⏳ Aguardando docs"
    st.markdown(f"**Chain RAG:** {chain_status}")

    if st.session_state.docs_processed:
        st.markdown(f"**Documentos:** {len(st.session_state.docs_processed)}")
        info = get_collection_info()
        st.markdown(
            f'<div class="metric-card">📦 Chunks indexados: <b>{info["chunks"]}</b></div>',
            unsafe_allow_html=True,
        )
        st.markdown("**Arquivos:**")
        for doc_name in st.session_state.docs_processed:
            st.markdown(f"  {get_file_icon(doc_name)} {doc_name}")
    else:
        st.info("Nenhum documento indexado ainda.")

    st.divider()

    # ── Controles ────────────────────────────────────────────
    st.markdown("### ⚙️ Controles")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("🗑️ Limpar Chat", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()
    with col2:
        if st.button("🔄 Reset Total", use_container_width=True):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()

    # ── Perguntas sugeridas ──────────────────────────────────
    st.divider()
    st.markdown("### 💡 Perguntas sugeridas")
    suggestions = [
        "O que é a nova economia espacial?",
        "Como os satélites monitoram o clima?",
        "Quais são as aplicações de IA na agricultura espacial?",
        "Como funciona o monitoramento de desastres por satélite?",
        "Quais missões da NASA exploram Marte?",
    ]
    for suggestion in suggestions:
        if st.button(suggestion, use_container_width=True, key=f"sug_{suggestion[:20]}"):
            st.session_state["pending_question"] = suggestion
            st.rerun()


# ══════════════════════════════════════════════════════════════
# ÁREA PRINCIPAL — CHAT
# ══════════════════════════════════════════════════════════════
st.markdown("""
<div class="main-header">
  <h1>🛸 SpaceRAG — Assistente Inteligente</h1>
  <p>Faça perguntas sobre seus documentos da Nova Economia Espacial</p>
</div>
""", unsafe_allow_html=True)

# ── Exibe histórico do chat ──────────────────────────────────
chat_container = st.container()

with chat_container:
    if not st.session_state.chat_history:
        st.markdown("""
        ### 👋 Bem-vindo ao SpaceRAG!

        Para começar:
        1. **Carregue documentos** na barra lateral (PDF, TXT ou DOCX)
        2. Clique em **Processar Documentos**
        3. **Faça suas perguntas** sobre os documentos carregados

        O sistema utilizará RAG (Retrieval-Augmented Generation) para
        buscar informações nos seus documentos antes de responder.
        """)
    else:
        for i, message in enumerate(st.session_state.chat_history):
            if message["role"] == "user":
                with st.chat_message("user", avatar="🧑‍🚀"):
                    st.markdown(message["content"])
            else:
                with st.chat_message("assistant", avatar="🤖"):
                    st.markdown(message["content"])
                    # Exibe fontes em expander
                    if message.get("sources"):
                        with st.expander("📚 Ver fontes consultadas", expanded=False):
                            st.markdown(
                                format_sources_markdown(message["sources"]),
                                unsafe_allow_html=True,
                            )

# ── Input de pergunta ────────────────────────────────────────
# Verifica se há pergunta pendente (dos botões de sugestão)
pending = st.session_state.pop("pending_question", None)
user_input = st.chat_input(
    placeholder="Faça uma pergunta sobre os documentos...",
    disabled=st.session_state.rag_chain is None,
)

# Usa a pergunta pendente ou a digitada pelo usuário
question = pending or user_input

if question:
    if not st.session_state.rag_chain:
        st.warning(
            "⚠️ Nenhum documento foi processado ainda. "
            "Carregue documentos na barra lateral primeiro."
        )
    else:
        # Adiciona pergunta ao histórico
        st.session_state.chat_history.append({
            "role": "user",
            "content": question,
        })

        # Exibe pergunta imediatamente
        with st.chat_message("user", avatar="🧑‍🚀"):
            st.markdown(question)

        # Gera resposta com spinner
        with st.chat_message("assistant", avatar="🤖"):
            with st.spinner("Buscando nos documentos e gerando resposta..."):
                try:
                    result = ask(st.session_state.rag_chain, question)
                    answer = result["answer"]
                    sources = result["sources"]
                except Exception as e:
                    logger.error(f"Erro na geração: {e}", exc_info=True)
                    answer = f"❌ Ocorreu um erro ao gerar a resposta: {str(e)}"
                    sources = []

            st.markdown(answer)

            if sources:
                with st.expander("📚 Ver fontes consultadas", expanded=False):
                    st.markdown(
                        format_sources_markdown(sources),
                        unsafe_allow_html=True,
                    )

        # Adiciona resposta ao histórico
        st.session_state.chat_history.append({
            "role": "assistant",
            "content": answer,
            "sources": sources,
        })

# ── Rodapé ───────────────────────────────────────────────────
st.divider()
st.markdown(
    "<p style='text-align:center; color: #555; font-size: 0.8rem;'>"
    "SpaceRAG · FIAP Global Solution 2026 · "
    "LangChain + ChromaDB + Llama 3 + Streamlit"
    "</p>",
    unsafe_allow_html=True,
)
