"""Shared LlamaIndex setup: the pgvector-backed vector store and the Ollama
embedding model, used by both ingestion (populate_vector_db.py) and
retrieval (run.py).
"""

from llama_index.core import Settings
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.vector_stores.postgres import PGVectorStore

try:
    from .config import Config
except ImportError:  # pragma: no cover - fallback for direct script execution
    from config import Config


def get_vector_store() -> PGVectorStore:
    """Build (or connect to) the Postgres-backed vector store.

    On first call this creates the table (and the pgvector extension/index)
    if it doesn't exist yet; on subsequent calls it just connects to it.
    """
    return PGVectorStore.from_params(
        host=Config.PG_HOST,
        port=Config.PG_PORT,
        database=Config.PG_DATABASE,
        user=Config.PG_USER,
        password=Config.PG_PASSWORD,
        table_name=Config.PG_TABLE_NAME,
        embed_dim=Config.EMBED_DIM,
    )


def configure_embedding_model() -> OllamaEmbedding:
    """Point LlamaIndex's global Settings at the Ollama embedding model and
    return it, so callers can also use it directly (e.g. to embed a query).
    """
    embed_model = OllamaEmbedding(
        model_name=Config.EMBEDDING_MODEL_NAME,
        base_url=Config.OLLAMA_BASE_URL,
        ollama_additional_kwargs=Config.OLLAMA_EMBEDDING_OPTIONS or {},
    )
    Settings.embed_model = embed_model
    return embed_model
