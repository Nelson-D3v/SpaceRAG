# 🛸 SpaceRAG — Assistente Inteligente para Nova Economia Espacial

> Sistema de Retrieval-Augmented Generation (RAG) para consulta inteligente
> de documentos sobre clima, satélites, agricultura espacial e exploração espacial.

## 📋 Sumário

- [Visão Geral](#visão-geral)
- [Arquitetura](#arquitetura)
- [Tecnologias](#tecnologias)
- [Instalação](#instalação)
- [Configuração](#configuração)
- [Execução](#execução)
- [Estrutura do Projeto](#estrutura-do-projeto)
- [Como Usar](#como-usar)

---

## Visão Geral

O SpaceRAG permite que usuários façam perguntas em linguagem natural sobre
documentos técnicos e científicos relacionados à nova economia espacial.
O sistema utiliza arquitetura RAG para garantir que as respostas sejam
baseadas nos documentos carregados, reduzindo alucinações do modelo.

## Arquitetura

```
Usuário → Streamlit → Document Loader → Text Splitter
                                              ↓
                              Embeddings (HuggingFace)
                                              ↓
                                         ChromaDB
                                              ↓
Pergunta → Query Embedding → Busca Semântica (top-k)
                                              ↓
                              Prompt Template + Contexto
                                              ↓
                              LLM (Llama 3 via Groq API)
                                              ↓
                                      Resposta + Fontes
```

## Tecnologias

| Componente       | Tecnologia                              |
|------------------|-----------------------------------------|
| Interface        | Streamlit 1.38                          |
| Framework RAG    | LangChain 0.2                           |
| Embeddings       | HuggingFace all-MiniLM-L6-v2            |
| Vector Store     | ChromaDB 0.5                            |
| LLM              | Llama 3.1 70B via Groq API (gratuito)   |
| PDF              | PyPDF 4.3                               |
| DOCX             | python-docx 1.1                         |

## Instalação

### Pré-requisitos

- Python 3.10+
- pip

### Passos

```bash
# 1. Clone ou extraia o projeto
cd space_rag

# 2. Crie um ambiente virtual
python -m venv venv
source venv/bin/activate      # Linux/Mac
# venv\Scripts\activate       # Windows

# 3. Instale as dependências
pip install -r requirements.txt
```

## Configuração

```bash
# Copie o arquivo de exemplo
cp .env.example .env

# Edite .env e adicione sua chave Groq
# Crie conta gratuita em: https://console.groq.com
nano .env
```

Conteúdo mínimo do `.env`:

```
GROQ_API_KEY=sua_chave_aqui
```

## Execução

```bash
streamlit run app.py
```

Acesse: http://localhost:8501

## Estrutura do Projeto

```
space_rag/
├── app.py                   # Interface principal Streamlit
├── config.py                # Configurações centralizadas
├── requirements.txt         # Dependências
├── .env.example             # Template de variáveis de ambiente
│
├── rag/
│   ├── __init__.py
│   └── rag_chain.py         # RetrievalQA Chain + LLM
│
├── embeddings/
│   ├── __init__.py
│   └── embedder.py          # HuggingFace Embeddings
│
├── vectorstore/
│   ├── __init__.py
│   └── chroma_store.py      # ChromaDB — indexação e busca
│
├── loaders/
│   ├── __init__.py
│   └── document_loader.py   # PDF, TXT, DOCX loaders
│
├── utils/
│   ├── __init__.py
│   └── helpers.py           # Funções auxiliares
│
├── docs/                    # Diretório para documentos de exemplo
└── README.md
```

## Como Usar

1. **Carregue documentos** — Use a barra lateral para fazer upload de PDFs,
   TXTs ou DOCXs sobre temas espaciais (relatórios NASA, artigos climáticos, etc.)

2. **Processe** — Clique em "Processar Documentos". O sistema irá:
   - Extrair o texto dos arquivos
   - Dividir em chunks de ~1000 caracteres
   - Gerar embeddings vetoriais
   - Armazenar no ChromaDB

3. **Pergunte** — Digite perguntas em linguagem natural. O sistema irá:
   - Converter sua pergunta em embedding
   - Buscar os 4 chunks mais relevantes
   - Gerar resposta fundamentada nos documentos
   - Mostrar as fontes consultadas

## Documentos Sugeridos

- [NASA Earth Observations](https://neo.gsfc.nasa.gov/)
- [INPE — Monitoramento Ambiental](https://www.inpe.br/)
- [ESA — Space Economy Reports](https://www.esa.int/)
- Relatórios IPCC sobre clima
- Artigos sobre agricultura de precisão via satélite

## Limitações

- Respostas limitadas ao conteúdo dos documentos carregados
- Modelo de embeddings pode ter desempenho reduzido em textos muito técnicos
- Limite de contexto do LLM (~32k tokens para Llama 3.1)

## Licença

Projeto acadêmico — FIAP Global Solution 2026
