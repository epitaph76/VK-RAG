import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    data_dir: Path = Path(os.getenv("DATA_DIR", "data"))
    root_path: str = os.getenv("ROOT_PATH", "")
    embedding_model: str = "intfloat/multilingual-e5-small"
    embedding_revision: str = "614241f622f53c4eeff9890bdc4f31cfecc418b3"
    laya_model: str = "convaiinnovations/laya-multilingual"
    laya_revision: str = "1720e3e3357cfe1e281542e223f8273b0890ca34"
    reranker_model: str = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
    reranker_revision: str = "1427fd652930e4ba29e8149678df786c240d8825"
    threads: int = int(os.getenv("MODEL_THREADS", "4"))
    chunk_tokens: int = 384
    overlap: int = 64
    gate_threshold: float = float(os.getenv("LAYA_KEEP_THRESHOLD", "0.8"))
    models_enabled: bool = os.getenv("MODELS_ENABLED", "1") == "1"
    api_key: str = os.getenv("SUB2API_API_KEY", "")
    api_url: str = os.getenv("SUB2API_BASE_URL", "").rstrip("/")
    api_model: str = os.getenv("SUB2API_MODEL", "gpt-5.6-luna")
    admin_password: str = os.getenv("ADMIN_PASSWORD", "")
    session_secret: str = os.getenv("SESSION_SECRET", "")
    secure_cookie: bool = os.getenv("SECURE_COOKIE", "0") == "1"
    ask_rate: int = int(os.getenv("ASK_RATE_PER_MINUTE", "5"))
    search_rate: int = int(os.getenv("SEARCH_RATE_PER_MINUTE", "30"))
    queue_size: int = int(os.getenv("PIPELINE_QUEUE_SIZE", "4"))
    source_limit: int = int(os.getenv("SOURCE_DOCUMENT_LIMIT", "100"))

    @property
    def database(self):
        return self.data_dir / "rag.sqlite3"
