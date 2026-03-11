"""
Database context for retriever system.

Provides database connection management for loading embeddings from database.
"""

import logging
import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from playhouse.postgres_ext import PostgresqlExtDatabase
from config_loader import get_database_config

env_path = Path(__file__).parent.parent / ".env"
if env_path.exists():
    load_dotenv(env_path)

logger = logging.getLogger(__name__)


def load_db_config(config_path: Optional[str] = None) -> dict:
    """Load database configuration from the tracked YAML config."""
    return get_database_config(config_path)


class RetrieverDbContext:
    """Database context for retriever tables."""
    
    def __init__(
        self,
        database: Optional[str] = None,
        host: Optional[str] = None,
        port: Optional[int] = None,
        user: Optional[str] = None,
        password: Optional[str] = None,
        config_path: Optional[str] = None,
    ):
        """
        Initialize database context.
        
        Args:
            database: Database name (if None, loads from config)
            host: Database host (if None, loads from config)
            port: Database port (if None, loads from config)
            user: Database user (if None, loads from config)
            password: Database password (if None, tries to get from env or config)
            config_path: Optional path to a database config YAML file
        """
        db_config = load_db_config(config_path)
        
        host = host or db_config.get("host", "localhost")
        port = port or db_config.get("port", 5432)
        database = database or db_config.get("name") or db_config.get("dbname", "rag_local")
        user = user or db_config.get("user", "postgres")
        
        if password is None:
            password_env = db_config.get("password_env", "PGPASSWORD_LOCAL")
            password = os.getenv(password_env) or os.getenv("PGPASSWORD_LOCAL") or os.getenv("PGPASSWORD")
        
        self.connect_params = {
            'host': host,
            'port': port,
            'database': database,
            'user': user,
            'password': password
        }
        
        self.database = PostgresqlExtDatabase(
            database,
            user=user,
            password=password,
            host=host,
            port=port
        )

        try:
            from database.entities import (
                PGVECTOR_AVAILABLE,
                TBChunk,
                TBDocument,
                TBEmbedding,
                TBPage,
                TBRun,
                register_vector,
            )

            self.runs = TBRun
            self.documents = TBDocument
            self.pages = TBPage
            self.chunks = TBChunk
            self.embeddings = TBEmbedding

            self._pgvector_available = PGVECTOR_AVAILABLE and register_vector is not None
            self._register_vector = register_vector

            for model in [TBRun, TBDocument, TBPage, TBChunk, TBEmbedding]:
                model._meta.database = self.database
        except ImportError as e:
            logger.warning(f"Failed to import database entities: {e}")
            self.runs = None
            self.documents = None
            self.pages = None
            self.chunks = None
            self.embeddings = None
            self._pgvector_available = False
            self._register_vector = None

    def connect(self):
        """Connect to database."""
        try:
            self.database.connect()
            if self._pgvector_available:
                try:
                    self._register_vector(self.database.connection())
                except Exception as register_error:
                    logger.warning(f"Failed to register pgvector adapter: {register_error}")
            logger.info(f"Connected to database: {self.database.database}")
        except Exception as e:
            logger.error(f"Failed to connect to database: {e}")
            raise
    
    def close(self):
        """Close database connection."""
        if not self.database.is_closed():
            self.database.close()
            logger.info("Database connection closed")
    
    def connection_context(self):
        """Context manager for database connection."""
        return self.database.connection_context()
