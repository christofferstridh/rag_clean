from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    # DB_URL: str = "postgresql://postgres:postgres@localhost/text_embeddings"
    # SHOULD_TRUNCATE: bool = True
    EMBEDDING_MODEL_NAME: str = "bge-m3:latest"
    REASONING_MODEL_NAME: str = "llama3.2:3b"
    OLLAMA_NUM_GPU: int = 1
    OLLAMA_KEEP_ALIVE: str = "10m"
    # för att ge default, skulle också kunna vara ex:
    # OLLAMA_EMBEDDING_OPTIONS: dict = {
    #     "temperature": 0.2,
    #     "num_ctx": 4096
    # }
    OLLAMA_EMBEDDING_OPTIONS: dict = None
    OLLAMA_REASONING_OPTIONS: dict = None
    # har varit nere på 1k o vänt vid perf problem, men 2k verkar funka bra nu
    OLLAMA_CONTEXT_SIZE: int = 2048
    # maximala antalet tokens (ord/bortdelar) som modellen får generera i ett och samma svar.
    OLLAMA_NUM_PREDICT: int = 128
