"""Load retrieval embeddings from the current schema and local artifacts."""

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from database.entities import TBChunk, TBDocument, TBEmbedding, TBPage
from retriever.db_context import RetrieverDbContext


logger = logging.getLogger(__name__)


class VectorLoader:
    """Load embedding vectors from database rows or local artifact files."""

    def __init__(
        self,
        db_context: Optional[RetrieverDbContext] = None,
        embeddings_dir: Optional[Path] = None,
        results_dir: Optional[Path] = None,
    ):
        self.db_context = db_context

        base_dir = Path(__file__).parent.parent
        self.embeddings_dir = embeddings_dir or (base_dir / "outputs" / "embeddings")
        self.results_dir = results_dir or (base_dir / "results")

    def load_from_database(
        self,
        chunk_id: Optional[str] = None,
        page_id: Optional[str] = None,
        embedding_mode: Optional[str] = None,
        doc_id: Optional[str] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """Load embeddings from the current `TBEmbedding` schema."""
        if self.db_context is None or self.db_context.embeddings is None:
            raise ValueError("Database context with bound TB* models is required")

        embeddings: Dict[str, Dict[str, Any]] = {}

        try:
            query = (
                self.db_context.embeddings
                .select(TBEmbedding, TBChunk, TBPage, TBDocument)
                .join(TBChunk)
                .join(TBPage)
                .switch(TBChunk)
                .join(TBDocument)
            )

            if chunk_id:
                query = query.where(TBEmbedding.chunk_id == chunk_id)
            if page_id:
                query = query.where(TBChunk.page_id == page_id)
            if embedding_mode:
                query = query.where(TBEmbedding.embedding_mode == embedding_mode)
            if doc_id:
                query = query.where(TBEmbedding.doc_id == doc_id)

            for row in query:
                record = self._build_database_record(row)
                record_id = record["record_id"]
                embeddings[record_id] = record

        except Exception as e:
            logger.error(f"Failed to load embeddings from database: {e}", exc_info=True)

        return embeddings

    def load_from_npz(
        self,
        record_id: Optional[str] = None,
        embedding_mode: Optional[str] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """Load embeddings from NPZ files."""
        embeddings: Dict[str, Dict[str, Any]] = {}

        if not self.embeddings_dir.exists():
            logger.warning(f"Embeddings directory does not exist: {self.embeddings_dir}")
            return embeddings

        for npz_path in self.embeddings_dir.glob("*.npz"):
            inferred_id = self._extract_record_id_from_filename(npz_path.stem)

            if record_id and record_id not in inferred_id and record_id not in npz_path.stem:
                continue

            try:
                file_data = self._load_embedding_file(npz_path)
                mode = self._infer_embedding_mode(file_data)
                if embedding_mode and mode != embedding_mode:
                    continue

                embedding_data = self._build_file_record(
                    record_id=inferred_id,
                    file_path=npz_path,
                    file_data=file_data,
                    metadata={"source": "npz"},
                )
                embeddings[inferred_id] = embedding_data
            except Exception as e:
                logger.warning(f"Failed to load {npz_path}: {e}")

        return embeddings

    def load_from_json(
        self,
        record_id: Optional[str] = None,
        embedding_mode: Optional[str] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """Load embeddings from JSON files."""
        embeddings: Dict[str, Dict[str, Any]] = {}

        if not self.results_dir.exists():
            logger.warning(f"Results directory does not exist: {self.results_dir}")
            return embeddings

        for json_path in self.results_dir.glob("*.json"):
            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    data = json.load(f)

                candidate = self._extract_embedding_payload_from_json(data)
                if candidate is None:
                    continue

                inferred_id = self._extract_record_id_from_json(data, candidate)
                if record_id and record_id not in inferred_id and record_id not in json_path.stem:
                    continue

                mode = self._infer_embedding_mode(candidate)
                if embedding_mode and mode != embedding_mode:
                    continue

                embedding_data = self._build_file_record(
                    record_id=inferred_id,
                    file_path=json_path,
                    file_data=candidate,
                    metadata={
                        "source": "json",
                        "timestamp": candidate.get("timestamp"),
                        "model": candidate.get("model"),
                        "image_size": candidate.get("image_size"),
                    },
                )
                embeddings[inferred_id] = embedding_data
            except Exception as e:
                logger.warning(f"Failed to load {json_path}: {e}")

        return embeddings

    def load_all(
        self,
        source: str = "auto",
        chunk_id: Optional[str] = None,
        page_id: Optional[str] = None,
        embedding_mode: Optional[str] = None,
        doc_id: Optional[str] = None,
        record_id: Optional[str] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """Load embeddings from the selected source."""
        embeddings: Dict[str, Dict[str, Any]] = {}

        if source == "auto":
            if self.db_context:
                try:
                    embeddings.update(
                        self.load_from_database(
                            chunk_id=chunk_id,
                            page_id=page_id,
                            embedding_mode=embedding_mode,
                            doc_id=doc_id,
                        )
                    )
                except Exception as e:
                    logger.warning(f"Failed to load from database: {e}")

            embeddings.update(self.load_from_npz(record_id=record_id or page_id or chunk_id, embedding_mode=embedding_mode))
            embeddings.update(self.load_from_json(record_id=record_id or page_id or chunk_id, embedding_mode=embedding_mode))
        elif source == "database":
            embeddings = self.load_from_database(
                chunk_id=chunk_id,
                page_id=page_id,
                embedding_mode=embedding_mode,
                doc_id=doc_id,
            )
        elif source == "npz":
            embeddings = self.load_from_npz(record_id=record_id or page_id or chunk_id, embedding_mode=embedding_mode)
        elif source == "json":
            embeddings = self.load_from_json(record_id=record_id or page_id or chunk_id, embedding_mode=embedding_mode)
        else:
            raise ValueError(f"Unknown source: {source}")

        logger.info(f"Loaded {len(embeddings)} embeddings from {source}")
        return embeddings

    def _build_database_record(self, row: TBEmbedding) -> Dict[str, Any]:
        """Convert a joined `TBEmbedding` row into retriever-ready data."""
        chunk = row.chunk_id
        page = getattr(chunk, "page_id", None)
        document = getattr(chunk, "doc_id", None)

        chunk_id = self._raw_fk_value(row, "chunk_id")
        page_id = self._raw_fk_value(chunk, "page_id") if chunk is not None else None
        chunk_doc_id = self._raw_fk_value(chunk, "doc_id") if chunk is not None else None
        doc_id = row.doc_id or chunk_doc_id

        record: Dict[str, Any] = {
            "record_id": chunk_id,
            "embedding_id": row.embedding_id,
            "chunk_id": chunk_id,
            "page_id": page_id,
            "page_number": getattr(page, "page_number", None),
            "doc_id": doc_id,
            "doc_name": getattr(document, "doc_name", None),
            "chunk_index": getattr(chunk, "chunk_index", None),
            "chunk_type": getattr(chunk, "chunk_type", None),
            "run_id": row.run_id,
            "vision_encoder": row.vision_encoder,
            "model_version": row.model_version,
            "text_encoder": row.text_encoder,
            "embedding_mode": row.embedding_mode,
            "embedding_dim": row.embedding_dim,
            "num_vectors": row.num_vectors,
            "vector_dim": row.vector_dim,
            "embedding_path": row.embedding_path,
            "storage_path": row.storage_path,
            "storage_format": row.storage_format,
            "file_size_bytes": row.file_size_bytes,
            "faiss_id": row.faiss_id,
            "qdrant_collection_name": row.qdrant_collection_name,
            "source_repo": row.source_repo,
            "data_source": row.data_source,
            "created_at": row.created_at,
            "updated_at": row.updated_at,
            "source": "database",
            "metadata": {},
        }

        pooled_embedding = self._to_float32_array(row.pooled_embedding_vector)
        if pooled_embedding is not None:
            record["pooled_embedding"] = pooled_embedding

        primary_path = row.storage_path if row.embedding_mode == "multi_vector" else row.embedding_path
        fallback_path = row.embedding_path if row.embedding_mode == "multi_vector" else row.storage_path

        file_data = self._load_embedding_file(primary_path) if primary_path else None
        if file_data is None and fallback_path and fallback_path != primary_path:
            file_data = self._load_embedding_file(fallback_path)

        if row.embedding_mode == "single_vector":
            vector = self._extract_single_vector(file_data)
            if vector is None:
                vector = pooled_embedding
            if vector is not None:
                record["embedding"] = vector
                record.setdefault("pooled_embedding", vector)
        elif row.embedding_mode == "multi_vector":
            token_embeddings = self._extract_token_embeddings(file_data)
            if token_embeddings is not None:
                record["token_embeddings"] = token_embeddings
                record["num_vectors"] = row.num_vectors or token_embeddings.shape[0]
                record["vector_dim"] = row.vector_dim or token_embeddings.shape[1]
                pooled_from_file = self._extract_pooled_vector(file_data)
                if pooled_from_file is None:
                    pooled_from_file = np.mean(token_embeddings, axis=0).astype(np.float32)
                record.setdefault("pooled_embedding", pooled_from_file)
            elif pooled_embedding is not None:
                record["pooled_embedding"] = pooled_embedding

        return record

    def _build_file_record(
        self,
        record_id: str,
        file_path: Path,
        file_data: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Convert a local artifact payload into retriever-ready data."""
        mode = self._infer_embedding_mode(file_data)
        record: Dict[str, Any] = {
            "record_id": record_id,
            "chunk_id": None,
            "page_id": record_id,
            "doc_id": None,
            "embedding_mode": mode,
            "embedding_dim": file_data.get("embedding_dim"),
            "num_vectors": file_data.get("num_tokens") or file_data.get("num_vectors"),
            "vector_dim": file_data.get("vector_dim"),
            "file_path": str(file_path),
            "source": metadata.get("source", "file") if metadata else "file",
            "metadata": metadata or {},
        }

        if mode == "single_vector":
            vector = self._extract_single_vector(file_data)
            if vector is not None:
                record["embedding"] = vector
                record["pooled_embedding"] = vector
                record["embedding_dim"] = record["embedding_dim"] or vector.shape[0]
        elif mode == "multi_vector":
            token_embeddings = self._extract_token_embeddings(file_data)
            if token_embeddings is not None:
                record["token_embeddings"] = token_embeddings
                pooled = self._extract_pooled_vector(file_data)
                if pooled is None:
                    pooled = np.mean(token_embeddings, axis=0).astype(np.float32)
                record["pooled_embedding"] = pooled
                record["embedding_dim"] = record["embedding_dim"] or token_embeddings.shape[1]
                record["num_vectors"] = record["num_vectors"] or token_embeddings.shape[0]
                record["vector_dim"] = record["vector_dim"] or token_embeddings.shape[1]

        return record

    def _load_embedding_file(self, file_path: Optional[str]) -> Optional[Dict[str, Any]]:
        """Load an embedding artifact from disk if it exists."""
        if not file_path:
            return None

        path = Path(file_path)
        if not path.exists():
            logger.warning(f"Embedding artifact not found: {path}")
            return None

        try:
            if path.suffix.lower() == ".npz":
                with np.load(path, allow_pickle=False) as npz_data:
                    return {key: npz_data[key] for key in npz_data.files}
            if path.suffix.lower() == ".npy":
                return {"embedding": np.load(path, allow_pickle=False)}
            if path.suffix.lower() == ".json":
                with open(path, "r", encoding="utf-8") as f:
                    payload = json.load(f)
                return payload if isinstance(payload, dict) else None
        except Exception as e:
            logger.warning(f"Failed to load embedding artifact {path}: {e}")

        return None

    def _extract_embedding_payload_from_json(self, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Return the embedding-bearing payload from a JSON artifact."""
        if "results" in data and data["results"]:
            candidate = data["results"][0]
            if isinstance(candidate, dict):
                return candidate
        if isinstance(data, dict):
            return data
        return None

    def _infer_embedding_mode(self, payload: Optional[Dict[str, Any]]) -> str:
        """Infer embedding mode from a payload."""
        if not payload:
            return "unknown"

        mode = payload.get("embedding_mode")
        if mode:
            return mode

        token_embeddings = payload.get("token_embeddings")
        if token_embeddings is not None:
            return "multi_vector"

        embedding = payload.get("embedding")
        if embedding is not None:
            array = self._to_float32_array(embedding)
            if array is not None and array.ndim == 2:
                return "multi_vector"
            return "single_vector"

        if payload.get("pooled_embedding") is not None:
            return "single_vector"

        return "unknown"

    def _extract_single_vector(self, payload: Optional[Dict[str, Any]]) -> Optional[np.ndarray]:
        """Extract a single vector from a payload."""
        if not payload:
            return None

        embedding = self._to_float32_array(payload.get("embedding"))
        if embedding is not None:
            if embedding.ndim == 1:
                return embedding
            if embedding.ndim == 2:
                return np.mean(embedding, axis=0).astype(np.float32)

        return self._extract_pooled_vector(payload)

    def _extract_token_embeddings(self, payload: Optional[Dict[str, Any]]) -> Optional[np.ndarray]:
        """Extract multi-vector token embeddings from a payload."""
        if not payload:
            return None

        token_embeddings = self._to_float32_array(payload.get("token_embeddings"))
        if token_embeddings is not None:
            return token_embeddings if token_embeddings.ndim == 2 else None

        embedding = self._to_float32_array(payload.get("embedding"))
        if embedding is not None and embedding.ndim == 2:
            return embedding

        return None

    def _extract_pooled_vector(self, payload: Optional[Dict[str, Any]]) -> Optional[np.ndarray]:
        """Extract the pooled vector from a payload."""
        if not payload:
            return None
        return self._to_float32_array(payload.get("pooled_embedding"))

    def _to_float32_array(self, value: Any) -> Optional[np.ndarray]:
        """Convert supported values into a float32 numpy array."""
        if value is None:
            return None

        if isinstance(value, np.ndarray):
            return value.astype(np.float32)

        if isinstance(value, (list, tuple)):
            return np.asarray(value, dtype=np.float32)

        if isinstance(value, str):
            stripped = value.strip()
            if not stripped:
                return None
            try:
                if stripped.startswith("[") and stripped.endswith("]"):
                    return np.asarray(json.loads(stripped), dtype=np.float32)
            except Exception:
                try:
                    return np.asarray(
                        [float(token) for token in stripped.strip("[]").split(",") if token],
                        dtype=np.float32,
                    )
                except Exception:
                    return None

        return None

    def _raw_fk_value(self, model: Any, field_name: str) -> Optional[str]:
        """Read the raw value for a Peewee foreign-key field."""
        if model is None:
            return None

        raw_attr = f"{field_name}_id"
        if hasattr(model, raw_attr):
            return getattr(model, raw_attr)

        value = getattr(model, field_name, None)
        if isinstance(value, str):
            return value
        if value is None:
            return None

        return getattr(value, field_name, None)

    def _extract_record_id_from_filename(self, filename: str) -> str:
        """Extract a stable identifier from a local artifact filename."""
        return filename

    def _extract_record_id_from_json(self, full_data: Dict[str, Any], result: Dict[str, Any]) -> str:
        """Extract a stable identifier from a JSON artifact."""
        for key in ("chunk_id", "page_id", "doc_id", "id"):
            if result.get(key):
                return str(result[key])

        input_file = full_data.get("input_file")
        if input_file:
            return Path(input_file).stem

        return Path(str(full_data.get("file_path", "unknown"))).stem or "unknown"
