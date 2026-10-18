"""Document ingestion and vector indexing script for Technical Datasheet RAG."""

import os
from pathlib import Path
from typing import List, Dict, Any
import chromadb
from chromadb.utils import embedding_functions
from rich.console import Console
from rich.table import Table

try:
    from pypdf import PdfReader
    PYPDF_AVAILABLE = True
except ImportError:
    PYPDF_AVAILABLE = False

from src.config import (
    DATA_DIR,
    CHROMA_DIR,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    EMBEDDING_MODEL_NAME,
)

console = Console()


def extract_text_from_file(file_path: Path) -> List[Dict[str, Any]]:
    """Extract text from .txt or .pdf files with page/section metadata."""
    documents = []

    if file_path.suffix.lower() == ".txt":
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
            documents.append({
                "text": content,
                "source": file_path.name,
                "page": 1
            })

    elif file_path.suffix.lower() == ".pdf":
        if not PYPDF_AVAILABLE:
            console.print("[yellow]Warning:[/yellow] pypdf is not installed. Skipping PDF files.")
            return []
        reader = PdfReader(str(file_path))
        for page_idx, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            if text.strip():
                documents.append({
                    "text": text,
                    "source": file_path.name,
                    "page": page_idx + 1
                })

    return documents


def recursive_character_chunking(
    text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP
) -> List[str]:
    """Split text into overlapping chunks respecting paragraph and line breaks."""
    if len(text) <= chunk_size:
        return [text.strip()] if text.strip() else []

    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]

        # Try to break at natural boundary (newline or period)
        if end < len(text):
            break_point = max(chunk.rfind("\n"), chunk.rfind(". "))
            if break_point > chunk_size // 2:
                chunk = chunk[:break_point + 1]
                end = start + break_point + 1

        cleaned = chunk.strip()
        if cleaned:
            chunks.append(cleaned)
        start = end - overlap

    return chunks


def build_vector_store() -> None:
    """Ingest documents from DATA_DIR, chunk them, embed, and store in ChromaDB."""
    console.print(f"[bold cyan]🔍 Scanning directory:[/bold cyan] {DATA_DIR}")

    if not DATA_DIR.exists():
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        console.print(f"[yellow]Created empty data directory at {DATA_DIR}. Please add technical docs.[/yellow]")
        return

    files = list(DATA_DIR.glob("*.txt")) + list(DATA_DIR.glob("*.pdf"))
    if not files:
        console.print("[red]No .txt or .pdf documents found in data directory![/red]")
        return

    all_chunks: List[str] = []
    all_metadatas: List[Dict[str, Any]] = []
    all_ids: List[str] = []

    for file_path in files:
        console.print(f"📄 Processing: [bold]{file_path.name}[/bold]")
        pages = extract_text_from_file(file_path)

        for page_info in pages:
            chunks = recursive_character_chunking(page_info["text"])
            for idx, chunk in enumerate(chunks):
                chunk_id = f"{file_path.stem}_p{page_info['page']}_c{idx}"
                all_chunks.append(chunk)
                all_metadatas.append({
                    "source": page_info["source"],
                    "page": page_info["page"],
                    "chunk_id": chunk_id
                })
                all_ids.append(chunk_id)

    console.print(f"\n[green]✓ Generated {len(all_chunks)} chunks from {len(files)} document(s).[/green]")

    # Setup local embedding function (100% free offline execution via sentence-transformers)
    console.print(f"⚙️  Initializing local embedding model: [bold]{EMBEDDING_MODEL_NAME}[/bold]...")
    emb_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL_NAME
    )

    # Initialize persistent ChromaDB client
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = client.get_or_create_collection(
        name="technical_datasheets",
        embedding_function=emb_fn,
        metadata={"hnsw:space": "cosine"}
    )

    # Upsert chunks into ChromaDB
    collection.upsert(
        documents=all_chunks,
        metadatas=all_metadatas,
        ids=all_ids
    )

    console.print(f"[bold green]✓ Successfully indexed {len(all_ids)} vectors into ChromaDB at:[/bold green] {CHROMA_DIR}\n")

    # Display summary table
    table = Table(title="Ingestion Summary")
    table.add_column("Document", style="cyan")
    table.add_column("Chunks", style="magenta")
    table.add_column("Embedding Model", style="green")

    for f in files:
        count = sum(1 for m in all_metadatas if m["source"] == f.name)
        table.add_row(f.name, str(count), EMBEDDING_MODEL_NAME)

    console.print(table)


if __name__ == "__main__":
    build_vector_store()
