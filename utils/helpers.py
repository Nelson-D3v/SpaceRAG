"""
utils/helpers.py
────────────────
Funções auxiliares utilizadas pela interface Streamlit e pelos
módulos de processamento.
"""

import os
import tempfile
import logging
from pathlib import Path
from typing import List

from config import SUPPORTED_EXTENSIONS

logger = logging.getLogger(__name__)


def save_uploaded_file(uploaded_file) -> str:
    """
    Salva um arquivo carregado pelo Streamlit em um diretório temporário.

    O Streamlit representa arquivos carregados como BytesIO — precisamos
    salvá-los em disco para que os loaders (PyPDF, python-docx) possam
    abri-los normalmente.

    Args:
        uploaded_file: Objeto retornado por st.file_uploader().

    Returns:
        Caminho absoluto do arquivo temporário salvo.
    """
    suffix = Path(uploaded_file.name).suffix.lower()
    with tempfile.NamedTemporaryFile(
        delete=False, suffix=suffix
    ) as tmp:
        tmp.write(uploaded_file.getvalue())
        tmp_path = tmp.name
    logger.info(f"Arquivo salvo temporariamente em: {tmp_path}")
    return tmp_path


def validate_file_extension(filename: str) -> bool:
    """
    Verifica se a extensão do arquivo é suportada.

    Args:
        filename: Nome do arquivo (com extensão).

    Returns:
        True se suportado, False caso contrário.
    """
    ext = Path(filename).suffix.lower()
    return ext in SUPPORTED_EXTENSIONS


def format_sources_markdown(sources: List[dict]) -> str:
    """
    Formata a lista de fontes para exibição em Markdown no Streamlit.

    Args:
        sources: Lista de dicts com chaves 'source', 'page', 'preview'.

    Returns:
        String formatada em Markdown.
    """
    if not sources:
        return "_Nenhuma fonte identificada._"

    lines = ["**📚 Fontes consultadas:**\n"]
    for i, src in enumerate(sources, 1):
        lines.append(
            f"**{i}. {src['source']}** "
            f"(pág. {src['page'] if src['page'] != '—' else 'N/A'})\n"
            f"> {src['preview']}\n"
        )
    return "\n".join(lines)


def cleanup_temp_file(path: str) -> None:
    """
    Remove um arquivo temporário do disco.

    Args:
        path: Caminho do arquivo a remover.
    """
    try:
        os.unlink(path)
        logger.debug(f"Arquivo temporário removido: {path}")
    except Exception as e:
        logger.warning(f"Não foi possível remover {path}: {e}")


def get_file_icon(filename: str) -> str:
    """
    Retorna um emoji representando o tipo do arquivo.

    Args:
        filename: Nome do arquivo.

    Returns:
        Emoji string.
    """
    ext = Path(filename).suffix.lower()
    icons = {
        ".pdf": "📄",
        ".txt": "📝",
        ".docx": "📃",
    }
    return icons.get(ext, "📁")
