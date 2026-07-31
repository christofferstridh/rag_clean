from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    # DB_URL: str = "postgresql://postgres:postgres@localhost/text_embeddings"
    # SHOULD_TRUNCATE: bool = True
    EMBEDDING_MODEL_NAME: str = "bge-m3:latest"
    REASONING_MODEL_NAME: str = "llama3.2:3b"
