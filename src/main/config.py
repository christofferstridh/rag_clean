from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    # DB_URL: str = "postgresql://postgres:postgres@localhost/text_embeddings"
    # SHOULD_TRUNCATE: bool = True
    EMBEDDING_MODEL_NAME: str = "bge-m3:latest"
    REASONING_MODEL_NAME: str = "llama3.2:3b"
    OLLAMA_NUM_GPU: int = 0
    OLLAMA_KEEP_ALIVE: str = "10m"
    OLLAMA_EMBEDDING_OPTIONS: dict = None
    OLLAMA_REASONING_OPTIONS: dict = None
    OLLAMA_CONTEXT_SIZE: int = 1024
    OLLAMA_NUM_PREDICT: int = 128
