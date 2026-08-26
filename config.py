class Config:
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200
    EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    LLM_MODEL = "openai/gpt-oss-20b"
    VECTORSTORE_TYPE = "chroma"
    RERANKER_TYPE = "flashrank"