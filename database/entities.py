"""
Database Entities for VLD-RAG System
Neon PostgreSQL Database Schema

This module defines ORM entities using Peewee for:
- Document management (documents, pages, chunks)
- Embedding storage (unified embedding table)
- Run tracking for experiments
"""

from datetime import datetime
from typing import Optional
import pytz

from peewee import (
    AutoField,
    BooleanField,
    CharField,
    DateTimeField,
    ForeignKeyField,
    IntegerField,
    Model,
    TextField,
    FloatField,
    BigIntegerField,
)
from playhouse.postgres_ext import BinaryJSONField
try:
    from pgvector.psycopg2 import register_vector
    PGVECTOR_AVAILABLE = True
except ImportError:
    PGVECTOR_AVAILABLE = False
    register_vector = None

# Import custom VectorField
from database.vector_field import VectorField


def get_kst_now():
    """Get current datetime in KST timezone as naive datetime (for TIMESTAMP WITHOUT TIME ZONE)."""
    kst = pytz.timezone('Asia/Seoul')
    kst_now = datetime.now(kst)
    # Convert to naive datetime (KST time without timezone info)
    # This will be stored as-is in TIMESTAMP WITHOUT TIME ZONE column
    return kst_now.replace(tzinfo=None)


class BaseModel(Model):
    """Base model with common fields"""
    
    class Meta:
        abstract = True


# ============================================================================
# Document Layer (tb_* tables for ACL experiments)
# ============================================================================

class TBRun(BaseModel):
    """Run tracking for ACL experiments"""
    
    class Meta:
        table_name = "tb_runs"
        indexes = (
            (("run_id",), True),
            (("source_repo",), False),
        )
    
    run_id = CharField(max_length=255, primary_key=True, column_name="run_id")
    run_name = CharField(max_length=500, null=True, column_name="run_name")
    note = TextField(null=True, column_name="note")
    created_by = CharField(max_length=255, null=True, column_name="created_by")
    is_active = BooleanField(default=True, column_name="is_active")
    source_repo = CharField(max_length=255, null=True, column_name="source_repo")
    metadata = BinaryJSONField(null=True, column_name="metadata")
    run_info = BinaryJSONField(null=True, column_name="run_info")
    created_at = DateTimeField(default=datetime.now, column_name="created_at")
    updated_at = DateTimeField(default=datetime.now, column_name="updated_at")


class TBDocument(BaseModel):
    """Document metadata and basic information"""
    
    class Meta:
        table_name = "tb_documents"
        indexes = (
            (("doc_id",), True),
            (("data_source",), False),
            (("source_repo",), False),
        )
    
    doc_id = CharField(max_length=255, primary_key=True, column_name="doc_id")
    doc_name = CharField(max_length=500, null=False, column_name="doc_name")
    doc_type = CharField(max_length=50, null=True, column_name="doc_type")
    source_path = TextField(null=True, column_name="source_path")
    data_source = CharField(max_length=255, null=True, column_name="data_source")
    data_source_path = TextField(null=True, column_name="data_source_path")
    source_repo = CharField(max_length=255, null=True, column_name="source_repo")
    total_pages = IntegerField(null=True, column_name="total_pages")
    total_chunks = IntegerField(null=True, column_name="total_chunks")
    created_at = DateTimeField(default=datetime.now, column_name="created_at")
    updated_at = DateTimeField(default=datetime.now, column_name="updated_at")
    metadata = BinaryJSONField(null=True, column_name="metadata")


class TBPage(BaseModel):
    """Page-level information within documents"""
    
    class Meta:
        table_name = "tb_pages"
        indexes = (
            (("page_id",), True),
            (("doc_id", "page_number"), True),
            (("parser_engine",), False),
        )
    
    page_id = CharField(max_length=255, primary_key=True, column_name="page_id")
    doc_id = ForeignKeyField(
        TBDocument,
        backref="pages",
        column_name="doc_id",
        to_field="doc_id",
        on_delete="CASCADE"
    )
    page_number = IntegerField(null=False, column_name="page_number")
    image_path = TextField(null=False, column_name="image_path")
    image_width = IntegerField(null=True, column_name="image_width")
    image_height = IntegerField(null=True, column_name="image_height")
    ocr_text = TextField(null=True, column_name="ocr_text")
    markdown_text = TextField(null=True, column_name="markdown_text")
    parser_engine = CharField(max_length=50, null=True, column_name="parser_engine")
    parser_version = CharField(max_length=50, null=True, column_name="parser_version")
    parsed_at = DateTimeField(null=True, column_name="parsed_at")
    created_at = DateTimeField(default=datetime.now, column_name="created_at")


class TBChunk(BaseModel):
    """Chunk-level information (crops/regions from pages)"""
    
    # Status values
    STATUS_CREATED = "created"      # Chunk created (only image path exists)
    STATUS_PARSING = "parsing"      # OCR/text parsing in progress
    STATUS_PARSED = "parsed"        # OCR/text parsing completed
    STATUS_ENCODING = "encoding"    # Embedding generation in progress
    STATUS_ENCODED = "encoded"      # Embedding generation completed
    STATUS_INDEXED = "indexed"      # Added to vector index (optional)
    STATUS_FAILED = "failed"        # Processing failed
    
    class Meta:
        table_name = "tb_chunks"
        indexes = (
            (("chunk_id",), True),
            (("page_id", "chunk_index"), True),
            (("chunk_type",), False),
            (("doc_id",), False),
            (("status",), False),
            (("qdrant_collection_name",), False),
        )
    
    chunk_id = CharField(max_length=255, primary_key=True, column_name="chunk_id")
    doc_id = ForeignKeyField(
        TBDocument,
        backref="chunks",
        column_name="doc_id",
        to_field="doc_id",
        on_delete="CASCADE"
    )
    page_id = ForeignKeyField(
        TBPage,
        backref="chunks",
        column_name="page_id",
        to_field="page_id",
        on_delete="CASCADE"
    )
    source_key = BinaryJSONField(null=True, column_name="source_key")
    chunk_index = IntegerField(null=False, column_name="chunk_index")
    bbox_x1 = IntegerField(null=True, column_name="bbox_x1")
    bbox_y1 = IntegerField(null=True, column_name="bbox_y1")
    bbox_x2 = IntegerField(null=True, column_name="bbox_x2")
    bbox_y2 = IntegerField(null=True, column_name="bbox_y2")
    crop_image_path = TextField(null=False, column_name="crop_image_path")
    crop_width = IntegerField(null=True, column_name="crop_width")
    crop_height = IntegerField(null=True, column_name="crop_height")
    chunk_type = CharField(max_length=50, null=True, column_name="chunk_type")
    ocr_text = TextField(null=True, column_name="ocr_text")
    markdown_text = TextField(null=True, column_name="markdown_text")
    parser_engine = CharField(max_length=50, null=True, column_name="parser_engine")
    block_id = CharField(max_length=255, null=True, column_name="block_id")
    status = CharField(
        max_length=50,
        null=True,
        default=STATUS_CREATED,
        column_name="status",
        help_text="Processing status: created, parsing, parsed, encoding, encoded, indexed, failed"
    )
    qdrant_collection_name = CharField(
        max_length=255,
        null=True,
        column_name="qdrant_collection_name",
        help_text="Qdrant collection name where this chunk is indexed (e.g., 'longdocurl-colpali', 'mmlongbench-colpali')"
    )
    created_at = DateTimeField(default=get_kst_now, column_name="created_at")


# ============================================================================
# Embedding Layer
# ============================================================================

class TBEmbedding(BaseModel):
    """Unified embedding table (supports all vision encoders)"""
    
    class Meta:
        table_name = "tb_embeddings"
        indexes = (
            (("embedding_id",), True),
            (("chunk_id", "vision_encoder"), True),
            (("vision_encoder",), False),
            (("embedding_mode",), False),
            (("faiss_id",), True),
            (("model_version",), False),
            (("qdrant_collection_name",), False),
            (("source_repo",), False),
            (("data_source",), False),
            (("doc_id",), False),
        )
    
    embedding_id = CharField(max_length=255, primary_key=True, column_name="embedding_id")
    chunk_id = ForeignKeyField(
        TBChunk,
        backref="embeddings",
        column_name="chunk_id",
        to_field="chunk_id",
        on_delete="CASCADE"
    )
    run_id = CharField(max_length=255, null=True, column_name="run_id")
    vision_encoder = CharField(
        max_length=100,
        null=False,
        column_name="vision_encoder",
        help_text="'siglip', 'colpali', 'omniembed', 'dse', etc."
    )
    model_version = CharField(max_length=100, null=False, column_name="model_version")
    text_encoder = CharField(max_length=100, null=True, column_name="text_encoder")
    embedding_mode = CharField(
        max_length=50,
        null=False,
        column_name="embedding_mode",
        help_text="'single_vector' or 'multi_vector'"
    )
    embedding_dim = IntegerField(null=True, column_name="embedding_dim", help_text="Single-vector dimension")
    num_vectors = IntegerField(null=True, column_name="num_vectors", help_text="Multi-vector token count")
    vector_dim = IntegerField(null=True, column_name="vector_dim", help_text="Multi-vector dimension")
    embedding_path = TextField(null=True, column_name="embedding_path", help_text="Single-vector file path")
    storage_path = TextField(null=True, column_name="storage_path", help_text="Multi-vector file path")
    storage_format = CharField(max_length=50, null=True, column_name="storage_format")
    file_size_bytes = BigIntegerField(null=True, column_name="file_size_bytes")
    faiss_id = IntegerField(null=True, unique=True, column_name="faiss_id", help_text="Faiss ID for single-vector")
    qdrant_collection_name = CharField(
        max_length=255,
        null=True,
        column_name="qdrant_collection_name",
        help_text="Qdrant collection name where this embedding is stored (e.g., 'longdocurl-colpali', 'mmlongbench-colpali')"
    )
    source_repo = CharField(
        max_length=255,
        null=True,
        column_name="source_repo",
        help_text="Source repository name (e.g., 'VLD-RAG')"
    )
    data_source = CharField(
        max_length=255,
        null=True,
        column_name="data_source",
        help_text="Data source name (e.g., 'mmlongbench', 'longdocurl')"
    )
    doc_id = CharField(
        max_length=255,
        null=True,
        column_name="doc_id",
        help_text="Document ID (denormalized from tb_chunks for faster queries)"
    )
    pooled_embedding_vector = VectorField(
        null=True,
        column_name="pooled_embedding_vector",
        help_text="Pooled embedding vector for pgvector (1152-dim, computed as mean of token embeddings for multi-vector mode)"
    )
    created_at = DateTimeField(default=get_kst_now, column_name="created_at")
    updated_at = DateTimeField(default=get_kst_now, column_name="updated_at")
