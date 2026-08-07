from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    # --- Postgres connection (used by PGVectorStore) ---
    PG_HOST: str = "localhost"
    PG_PORT: str = "5432"
    PG_DATABASE: str = "text_embeddings"
    PG_USER: str = "postgres"
    PG_PASSWORD: str = "postgres"
    # New table for the LlamaIndex-managed vector store. Kept separate from any
    # old hand-rolled `text_embeddings` table since the schema (node_id, metadata_,
    # etc.) is different and requires re-ingestion.
    PG_TABLE_NAME: str = "sentence_windows"

    # --- Ollama ---
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    EMBEDDING_MODEL_NAME: str = "bge-m3:latest"
    EMBED_DIM: int = 1024  # bge-m3 dense embedding dimension
    REASONING_MODEL_NAME: str = "llama3.2:3b"
    OLLAMA_NUM_GPU: int = 1
    OLLAMA_KEEP_ALIVE: str = "10m"
    OLLAMA_EMBEDDING_OPTIONS: dict = None
    OLLAMA_REASONING_OPTIONS: dict = None
    # har varit nere på 1k o vänt vid perf problem, men 2k verkar funka bra nu
    OLLAMA_CONTEXT_SIZE: int = 2048
    # maximala antalet tokens (ord/bortdelar) som modellen får generera i ett och samma svar.
    OLLAMA_NUM_PREDICT: int = 256

    # --- Retrieval ---
    # Sentences captured on either side of a matched sentence (replaces group_window_size).
    WINDOW_SIZE: int = 1
    # 0 = max diversity between results, 1 = max relevance/similarity (replaces the
    # hand-rolled is_unique_to_window near-duplicate filter).
    MMR_THRESHOLD: float = 0.5
