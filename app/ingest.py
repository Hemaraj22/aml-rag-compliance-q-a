"""Load FFIEC BSA/AML PDFs, semantically chunk them, and index them in ChromaDB.

Usage:  python -m app.ingest
"""
import hashlib
import logging
import re
from collections import Counter
from datetime import date
from pathlib import Path

from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import get_settings

log = logging.getLogger(__name__)


def detect_edition(page_texts: list[str]) -> int | None:
    """Edition year from page headers ("Examination Manual 12 February 2021"); if a file has no
    dated headers, fall back to the newest year mentioned anywhere in it."""
    headers = [
        m.group(1)
        for t in page_texts
        if (m := re.search(r"Examination Manual\s+\d+\s+[^\n]*?\b((?:19|20)\d{2})\b", t))
    ]
    if headers:
        return int(Counter(headers).most_common(1)[0][0])
    years = [int(y) for t in page_texts for y in re.findall(r"\b(?:19|20)\d{2}\b", t)]
    years = [y for y in years if y <= date.today().year]
    return max(years) if years else None


def load_pdfs(data_dir: Path) -> list[Document]:
    pdfs = sorted(data_dir.glob("**/*.pdf"))
    if not pdfs:
        raise FileNotFoundError(
            f"No PDFs in {data_dir}. Download FFIEC BSA/AML manual sections first (see README)."
        )
    docs: list[Document] = []
    for pdf in pdfs:
        pages = PyPDFLoader(str(pdf)).load()
        edition = detect_edition([p.page_content for p in pages])
        for p in pages:
            p.metadata = {
                "source": pdf.name,
                "section": pdf.stem.replace("_", " "),
                "page": int(p.metadata.get("page", 0)) + 1,  # 1-based for citations
                "edition": edition or 0,  # Chroma metadata can't be None
            }
        docs.extend(pages)
        log.info("Loaded %s (%d pages, edition %s)", pdf.name, len(pages), edition)
    return docs


def chunk_documents(docs: list[Document], embeddings: OpenAIEmbeddings, max_chars: int) -> list[Document]:
    """Semantic chunking (split where embedding similarity drops), with a size cap."""
    semantic = SemanticChunker(embeddings, breakpoint_threshold_type="percentile", breakpoint_threshold_amount=90)
    fallback = RecursiveCharacterTextSplitter(chunk_size=max_chars, chunk_overlap=200)

    chunks: list[Document] = []
    for doc in docs:
        if not doc.page_content.strip():
            continue
        for chunk in semantic.split_documents([doc]):
            if len(chunk.page_content) > max_chars:
                chunks.extend(fallback.split_documents([chunk]))
            else:
                chunks.append(chunk)

    for c in chunks:
        key = f"{c.metadata['source']}|{c.metadata['page']}|{c.page_content}"
        c.metadata["chunk_id"] = hashlib.sha1(key.encode()).hexdigest()[:16]
    return chunks


def build_index() -> int:
    s = get_settings()
    embeddings = OpenAIEmbeddings(model=s.embedding_model)
    docs = load_pdfs(Path(s.data_dir))
    chunks = chunk_documents(docs, embeddings, s.max_chunk_chars)

    store = Chroma(
        collection_name=s.collection_name,
        embedding_function=embeddings,
        persist_directory=s.chroma_dir,
        collection_metadata={"hnsw:space": "cosine"},
    )
    store.reset_collection()  # full rebuild keeps the index consistent with data/raw

    # de-duplicate identical chunks, then add in batches
    unique = {c.metadata["chunk_id"]: c for c in chunks}
    ids, batch = list(unique), list(unique.values())
    for i in range(0, len(batch), 200):
        store.add_documents(batch[i : i + 200], ids=ids[i : i + 200])

    log.info("Indexed %d chunks from %d pages", len(batch), len(docs))
    return len(batch)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    build_index()
