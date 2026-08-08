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
    # har varit nere på 1k o vänt vid perf problem, men 8k verkar funka bra nu, dock segt - kör 4k
    OLLAMA_CONTEXT_SIZE: int = 1024 * 2**2
    # maximala antalet tokens (ord/bortdelar) som modellen får generera i ett och samma svar. 256 verkar oftast räcka, men 512 kan ge mer utförliga svar.
    OLLAMA_NUM_PREDICT: int = 128 * 2**1

    # --- Retrieval ---
    # denna trimmas ner senare i kod...
    EMBEDDING_NUM_MATCHES: int = 5
    # ...genom att den här används, om ett svar uppfyller denna så används bara den o i annat fall fyller man på för att uppnå total score
    EMBEDDING_SCORE_TARGET: float = 0.55
    # ... o max så här många till llm
    LLM_MAX_CONTEXT_NODES: int = 3
