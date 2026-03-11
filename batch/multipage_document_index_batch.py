"""Batch index multipage documents into the current `TB*` schema.

Supported inputs:
- a single PDF file
- a directory of page images for one document
- a directory containing multiple PDF files and/or page-image subdirectories

The batch flow can:
1. render PDFs to page images (when PyMuPDF is installed)
2. parse each page with `PaddleOCRParser`
3. save document/page/chunk rows into PostgreSQL via the current Peewee models
4. optionally generate ColPali chunk embeddings and persist them in `tb_embeddings`
"""

from __future__ import annotations

import argparse
import logging
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Sequence

from PIL import Image

from config_loader import get_data_config, get_model_config, resolve_repo_path
from parser.engines import PaddleOCRParser
from retriever import ColPaliVisionRetriever, RetrieverDbContext

try:
    import fitz  # PyMuPDF

    PYMUPDF_AVAILABLE = True
except ImportError:
    fitz = None
    PYMUPDF_AVAILABLE = False


logger = logging.getLogger(__name__)

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}


@dataclass
class DocumentInput:
    """Normalized input description for a single document indexing job."""

    doc_id: str
    source_type: str  # "pdf" or "page_images"
    source_path: Path


def configure_logging(verbose: bool = False) -> None:
    """Configure process-wide logging."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s - %(levelname)s - %(message)s")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Batch index multipage documents into VLD-RAG tables")
    parser.add_argument(
        "input_path",
        nargs="?",
        help="Optional PDF, page-image directory, or directory of documents. If omitted, configured data paths are scanned.",
    )
    parser.add_argument("--doc-id", help="Explicit document ID when indexing a single file or page-image directory")
    parser.add_argument("--data-source", default="sample", help="Logical dataset/source name stored in the DB")
    parser.add_argument("--device", default="cpu", help="Parser/encoder device, e.g. cpu or cuda")
    parser.add_argument("--limit", type=int, default=None, help="Maximum number of documents to process")
    parser.add_argument(
        "--replace-existing",
        action="store_true",
        help="Delete and recreate an existing document before indexing",
    )
    parser.add_argument(
        "--with-embeddings",
        action="store_true",
        help="Generate ColPali chunk embeddings and save them to tb_embeddings",
    )
    parser.add_argument(
        "--embedding-mode",
        choices=("single_vector", "multi_vector"),
        default="multi_vector",
        help="Embedding storage mode when --with-embeddings is enabled",
    )
    parser.add_argument(
        "--colpali-model",
        default=None,
        help="Optional local path or Hugging Face model ID for ColPali",
    )
    parser.add_argument(
        "--database-config",
        default=None,
        help="Optional path to a database YAML config",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    return parser.parse_args()


def main() -> None:
    """Run the multipage document indexing batch."""
    args = parse_args()
    configure_logging(args.verbose)

    data_config = get_data_config()
    paths_cfg = data_config.get("paths", {})

    page_images_root = resolve_repo_path(paths_cfg.get("page_images_dir")) or Path("data/page_images").resolve()
    chunks_root = resolve_repo_path(paths_cfg.get("chunks_dir")) or Path("data/chunks").resolve()
    embeddings_root = resolve_repo_path(paths_cfg.get("embeddings_dir")) or Path("data/embeddings").resolve()
    raw_documents_root = resolve_repo_path(paths_cfg.get("raw_documents_dir")) or Path("data/raw").resolve()

    document_inputs = discover_document_inputs(
        input_path=args.input_path,
        doc_id=args.doc_id,
        raw_documents_root=raw_documents_root,
        page_images_root=page_images_root,
        limit=args.limit,
    )
    if not document_inputs:
        raise SystemExit("No documents found to index.")

    db = RetrieverDbContext(config_path=args.database_config)
    parser = PaddleOCRParser(device=args.device)
    parser.initialize()

    embedder = None
    if args.with_embeddings:
        embedder = build_colpali_embedder(model_override=args.colpali_model, device=args.device)

    db.connect()
    try:
        with db.connection_context():
            run_id = create_or_update_run(db, args)
            total_docs = 0
            for document_input in document_inputs:
                index_document(
                    db=db,
                    parser=parser,
                    embedder=embedder,
                    run_id=run_id,
                    document_input=document_input,
                    data_source=args.data_source,
                    page_images_root=page_images_root,
                    chunks_root=chunks_root,
                    embeddings_root=embeddings_root,
                    replace_existing=args.replace_existing,
                    with_embeddings=args.with_embeddings,
                    embedding_mode=args.embedding_mode,
                )
                total_docs += 1

        logger.info(f"Indexed {total_docs} document(s) successfully.")
    finally:
        db.close()


def discover_document_inputs(
    input_path: Optional[str],
    doc_id: Optional[str],
    raw_documents_root: Path,
    page_images_root: Path,
    limit: Optional[int] = None,
) -> List[DocumentInput]:
    """Discover documents from explicit input or configured data paths."""
    candidates: List[DocumentInput] = []

    if input_path:
        candidates.extend(_discover_from_path(Path(input_path), explicit_doc_id=doc_id))
    else:
        if raw_documents_root.exists():
            candidates.extend(_discover_from_path(raw_documents_root))
        if page_images_root.exists():
            candidates.extend(_discover_from_path(page_images_root))

    deduped: List[DocumentInput] = []
    seen = set()
    for candidate in candidates:
        if candidate.doc_id in seen:
            continue
        deduped.append(candidate)
        seen.add(candidate.doc_id)

    return deduped[:limit] if limit is not None else deduped


def _discover_from_path(path: Path, explicit_doc_id: Optional[str] = None) -> List[DocumentInput]:
    """Normalize an input path into one or more document jobs."""
    path = path.expanduser().resolve()
    if not path.exists():
        return []

    if path.is_file():
        if path.suffix.lower() == ".pdf":
            return [DocumentInput(doc_id=explicit_doc_id or path.stem, source_type="pdf", source_path=path)]
        if path.suffix.lower() in IMAGE_SUFFIXES:
            return [DocumentInput(doc_id=explicit_doc_id or path.stem, source_type="page_images", source_path=path.parent)]
        return []

    direct_images = sorted([child for child in path.iterdir() if child.is_file() and child.suffix.lower() in IMAGE_SUFFIXES])
    direct_pdfs = sorted([child for child in path.iterdir() if child.is_file() and child.suffix.lower() == ".pdf"])
    subdirs = sorted([child for child in path.iterdir() if child.is_dir()])

    if direct_images and not direct_pdfs and not subdirs:
        return [DocumentInput(doc_id=explicit_doc_id or path.name, source_type="page_images", source_path=path)]

    jobs: List[DocumentInput] = []
    for pdf_path in direct_pdfs:
        jobs.append(DocumentInput(doc_id=pdf_path.stem, source_type="pdf", source_path=pdf_path))
    for subdir in subdirs:
        subdir_images = any(child.is_file() and child.suffix.lower() in IMAGE_SUFFIXES for child in subdir.iterdir())
        if subdir_images:
            jobs.append(DocumentInput(doc_id=subdir.name, source_type="page_images", source_path=subdir))
    return jobs


def build_colpali_embedder(model_override: Optional[str], device: str) -> ColPaliVisionRetriever:
    """Construct the optional ColPali embedder from config or CLI override."""
    model_config = get_model_config()
    colpali_cfg = model_config.get("retrieval", {}).get("dense", {}).get("colpali", {})
    model_name = model_override
    if not model_name:
        local_path = colpali_cfg.get("local_path")
        resolved_local = resolve_repo_path(local_path)
        if resolved_local and resolved_local.exists():
            model_name = str(resolved_local)
        else:
            model_name = colpali_cfg.get("model_id", "vidore/colpali-v1.2")

    logger.info(f"Using ColPali model source: {model_name}")
    return ColPaliVisionRetriever(model_name=model_name, device=device, source="database")


def create_or_update_run(db: RetrieverDbContext, args: argparse.Namespace) -> str:
    """Create a run record for this batch execution."""
    run_id = f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    if db.runs is None:
        return run_id

    db.runs.create(
        run_id=run_id,
        run_name="multipage_document_index_batch",
        note="Batch indexing run for multipage documents",
        created_by="batch.multipage_document_index_batch",
        is_active=True,
        source_repo="VLD-RAG",
        metadata={
            "data_source": args.data_source,
            "with_embeddings": args.with_embeddings,
            "embedding_mode": args.embedding_mode,
        },
        run_info={
            "input_path": args.input_path,
            "replace_existing": args.replace_existing,
            "device": args.device,
        },
    )
    return run_id


def index_document(
    db: RetrieverDbContext,
    parser: PaddleOCRParser,
    embedder: Optional[ColPaliVisionRetriever],
    run_id: str,
    document_input: DocumentInput,
    data_source: str,
    page_images_root: Path,
    chunks_root: Path,
    embeddings_root: Path,
    replace_existing: bool,
    with_embeddings: bool,
    embedding_mode: str,
) -> None:
    """Index a single multipage document into the current schema."""
    logger.info(f"Indexing document `{document_input.doc_id}` from {document_input.source_path}")

    existing_document = db.documents.get_or_none(db.documents.doc_id == document_input.doc_id)
    if existing_document is not None:
        if not replace_existing:
            logger.info(f"Skipping existing document `{document_input.doc_id}`. Use --replace-existing to rebuild it.")
            return
        logger.info(f"Deleting existing document `{document_input.doc_id}` before re-indexing.")
        existing_document.delete_instance(recursive=True, delete_nullable=True)

    page_image_paths = materialize_page_images(document_input, page_images_root)
    if not page_image_paths:
        logger.warning(f"No page images found for document `{document_input.doc_id}`. Skipping.")
        return

    source_path_string = path_for_storage(document_input.source_path)
    doc_type = "pdf" if document_input.source_type == "pdf" else "page_images"

    document = db.documents.create(
        doc_id=document_input.doc_id,
        doc_name=document_input.doc_id,
        doc_type=doc_type,
        source_path=source_path_string,
        data_source=data_source,
        data_source_path=source_path_string,
        source_repo="VLD-RAG",
        total_pages=len(page_image_paths),
        total_chunks=0,
        metadata={"source_type": document_input.source_type},
    )

    total_chunks = 0
    for page_number, image_path in enumerate(page_image_paths, start=1):
        image = Image.open(image_path).convert("RGB")
        page_id = f"{document_input.doc_id}_page_{page_number:04d}"

        page_parse = parser.parse_page(
            doc_id=document_input.doc_id,
            page_no=page_number - 1,
            image=image,
            image_path=path_for_storage(image_path),
        )
        elements = parser.normalize_to_rag_elements(page_parse)

        page_markdown_text = "\n\n".join(filter(None, [parser._serialize_block_markdown(block) for block in page_parse.blocks]))
        page_ocr_text = "\n\n".join(filter(None, [block.text for block in page_parse.blocks if block.text]))

        db.pages.create(
            page_id=page_id,
            doc_id=document.doc_id,
            page_number=page_number,
            image_path=path_for_storage(image_path),
            image_width=image.width,
            image_height=image.height,
            ocr_text=page_ocr_text or None,
            markdown_text=page_markdown_text or None,
            parser_engine="paddle_ocr",
            parser_version="3.0.0",
            parsed_at=datetime.now(),
        )

        chunk_dir = chunks_root / document_input.doc_id / f"page_{page_number:04d}"
        chunk_ids = parser.save_to_tb_chunks(
            elements=elements,
            doc_id=document.doc_id,
            page_id=page_id,
            source_key={"doc_id": document.doc_id, "page_number": page_number},
            page_image=image,
            chunk_image_dir=str(chunk_dir),
        )
        total_chunks += len(chunk_ids)

        if with_embeddings and embedder is not None:
            save_chunk_embeddings(
                db=db,
                embedder=embedder,
                run_id=run_id,
                doc_id=document.doc_id,
                chunk_ids=chunk_ids,
                embeddings_root=embeddings_root,
                embedding_mode=embedding_mode,
            )

    document.total_chunks = total_chunks
    document.save()
    logger.info(f"Finished document `{document_input.doc_id}` with {document.total_pages} pages and {total_chunks} chunks.")


def materialize_page_images(document_input: DocumentInput, page_images_root: Path) -> List[Path]:
    """Return rendered or pre-existing page images for one document."""
    if document_input.source_type == "page_images":
        return sorted_image_paths(document_input.source_path)

    if document_input.source_type != "pdf":
        return []

    if not PYMUPDF_AVAILABLE:
        raise RuntimeError("PyMuPDF is required to render PDF inputs. Install `pymupdf` or provide page images instead.")

    output_dir = page_images_root / document_input.doc_id
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Rendering PDF `{document_input.source_path}` to page images in `{output_dir}`")
    image_paths: List[Path] = []
    with fitz.open(document_input.source_path) as pdf_document:
        for page_index in range(len(pdf_document)):
            page = pdf_document.load_page(page_index)
            pix = page.get_pixmap(dpi=300)
            output_path = output_dir / f"page_{page_index + 1:04d}.png"
            pix.save(output_path)
            image_paths.append(output_path)
    return image_paths


def sorted_image_paths(directory: Path) -> List[Path]:
    """Return page image files in a stable page-number order."""
    image_paths = [child for child in directory.iterdir() if child.is_file() and child.suffix.lower() in IMAGE_SUFFIXES]
    return sorted(image_paths, key=_page_sort_key)


def _page_sort_key(path: Path) -> tuple[int, str]:
    """Sort paths by numeric page hint when available."""
    match = re.search(r"(\d+)", path.stem)
    page_num = int(match.group(1)) if match else 10**9
    return page_num, path.name.lower()


def save_chunk_embeddings(
    db: RetrieverDbContext,
    embedder: ColPaliVisionRetriever,
    run_id: str,
    doc_id: str,
    chunk_ids: Sequence[str],
    embeddings_root: Path,
    embedding_mode: str,
) -> None:
    """Generate and persist ColPali chunk embeddings for saved chunks."""
    vision_encoder = "colpali"
    model_version = str(embedder.model_name)

    for chunk_id in chunk_ids:
        chunk = db.chunks.get_or_none(db.chunks.chunk_id == chunk_id)
        if chunk is None:
            logger.warning(f"Chunk `{chunk_id}` was not found for embedding generation.")
            continue
        if not chunk.crop_image_path:
            logger.warning(f"Chunk `{chunk_id}` has no crop image path. Skipping embedding generation.")
            continue

        crop_path = Path(chunk.crop_image_path)
        if not crop_path.exists():
            logger.warning(f"Chunk crop image does not exist: {crop_path}")
            continue

        image = Image.open(crop_path).convert("RGB")
        encoded = embedder.encode_image(image, embedding_mode=embedding_mode)

        embedding_dir = embeddings_root / doc_id
        embedding_dir.mkdir(parents=True, exist_ok=True)
        embedding_file = embedding_dir / f"{chunk_id}.npz"

        if embedding_mode == "multi_vector":
            token_embeddings = encoded["token_embeddings"]
            pooled_embedding = encoded["pooled_embedding"]
            import numpy as np

            np.savez_compressed(
                embedding_file,
                token_embeddings=token_embeddings,
                pooled_embedding=pooled_embedding,
            )
            embedding_path_value = None
            storage_path_value = path_for_storage(embedding_file)
            embedding_dim = token_embeddings.shape[1]
            num_vectors = int(encoded.get("num_tokens", token_embeddings.shape[0]))
            vector_dim = token_embeddings.shape[1]
            pooled_vector = pooled_embedding
        else:
            vector = encoded["embedding"]
            import numpy as np

            np.savez_compressed(embedding_file, embedding=vector, pooled_embedding=vector)
            embedding_path_value = path_for_storage(embedding_file)
            storage_path_value = None
            embedding_dim = vector.shape[0]
            num_vectors = 1
            vector_dim = vector.shape[0]
            pooled_vector = encoded.get("pooled_embedding", vector)

        embedding_id = f"{chunk_id}__{vision_encoder}"
        defaults = {
            "run_id": run_id,
            "vision_encoder": vision_encoder,
            "model_version": model_version,
            "text_encoder": None,
            "embedding_mode": embedding_mode,
            "embedding_dim": embedding_dim,
            "num_vectors": num_vectors,
            "vector_dim": vector_dim,
            "embedding_path": embedding_path_value,
            "storage_path": storage_path_value,
            "storage_format": "npz",
            "file_size_bytes": embedding_file.stat().st_size,
            "faiss_id": None,
            "qdrant_collection_name": None,
            "source_repo": "VLD-RAG",
            "data_source": None,
            "doc_id": doc_id,
            "pooled_embedding_vector": pooled_vector,
        }

        record, created = db.embeddings.get_or_create(
            embedding_id=embedding_id,
            defaults={"chunk_id": chunk.chunk_id, **defaults},
        )
        if not created:
            for key, value in defaults.items():
                setattr(record, key, value)
            record.chunk_id = chunk.chunk_id
            record.updated_at = datetime.now()
            record.save()


def path_for_storage(path: Path) -> str:
    """Prefer repository-relative path strings when possible."""
    resolved = path.resolve()
    try:
        project_root = Path(__file__).resolve().parent.parent
        return str(resolved.relative_to(project_root))
    except ValueError:
        return str(resolved)


if __name__ == "__main__":
    main()
