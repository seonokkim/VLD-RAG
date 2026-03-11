# Retriever Module

This directory contains retrieval-related components for `VLD-RAG`, including:

- `bm25_retriever.py`
- `colpali_vision_retriever.py`
- `db_context.py`
- `hybrid_retriever.py`
- `hyde.py`
- `scorer.py`
- `text_embedding_retriever.py`
- `vector_loader.py`

## Database Settings

The database-facing entrypoint in this directory is `retriever/db_context.py`.

`RetrieverDbContext` loads database settings from the tracked repository config and then resolves the password from environment variables.

### Default config file

Non-secret connection settings are stored in:

- `configs/database.yml`

Current default structure:

```yaml
database:
  host: localhost
  port: 5432
  name: rag_local
  user: postgres
  password_env: PGPASSWORD_LOCAL
```

### Secret handling

Do not store database passwords directly in `configs/database.yml`.

Instead, set the password in an environment variable whose name matches `password_env`.

Current default:

- `PGPASSWORD_LOCAL`

If `.env` exists at the repository root, `retriever/db_context.py` will load it automatically before resolving the password.

Starter template:

```env
# See `.env.example` at the repository root.
PGPASSWORD_LOCAL=your_database_password
```

### Resolution order

`RetrieverDbContext(...)` resolves settings in this order:

1. Explicit constructor arguments
2. Config values from `configs/database.yml`
3. Environment variable referenced by `password_env`
4. Built-in fallback defaults

For the password specifically, the current lookup order is:

1. explicit `password=...`
2. env var named by `password_env`
3. `PGPASSWORD_LOCAL`
4. `PGPASSWORD`

## Usage

Basic example:

```python
from retriever.db_context import RetrieverDbContext

db = RetrieverDbContext()
db.connect()

with db.connection_context():
    count = db.embeddings.select().count()
    print(count)

db.close()
```

Using a custom config path:

```python
from retriever.db_context import RetrieverDbContext

db = RetrieverDbContext(config_path="configs/database.yml")
```

Overriding values directly:

```python
from retriever.db_context import RetrieverDbContext

db = RetrieverDbContext(
    host="localhost",
    port=5432,
    database="rag_local",
    user="postgres",
)
```

## Related Files

- `config_loader.py`: shared YAML config loading helpers
- `configs/database.yml`: non-secret database defaults
- `database/entities.py`: active Peewee schema bound by `RetrieverDbContext`
- `database/vector_field.py`: pgvector field support
- `vector_loader.py`: loads embeddings from database rows or local artifacts

## Notes

- `RetrieverDbContext.connect()` attempts to register the pgvector adapter when available.
- `vector_loader.py` can load embeddings from the database, `.npz` artifacts, or `.json` artifacts.
- `text_embedding_retriever.py` provides dense text retrieval over stored text embeddings using an injected query embedder and the current `TBEmbedding` schema.
- `hybrid_retriever.py` can fuse sparse BM25 results with dense retrieval results, including HyDE-expanded dense queries.
- The retriever layer still uses PostgreSQL primarily as a schema-backed storage and loading layer, not yet as a full vector-store service with collection lifecycle and hybrid SQL search management.
