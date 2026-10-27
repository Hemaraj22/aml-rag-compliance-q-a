from functools import lru_cache

from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict

# OpenAI and LangSmith clients read their keys from os.environ, not from Settings
load_dotenv()


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    openai_api_key: str = ""
    llm_model: str = "gpt-4o-mini"
    embedding_model: str = "text-embedding-3-small"

    chroma_dir: str = "./chroma_db"
    collection_name: str = "ffiec_bsa_aml"
    data_dir: str = "./data/raw"

    top_k: int = 5
    min_relevance: float = 0.3
    max_chunk_chars: int = 2000

    # Hybrid retrieval (vector + BM25 keyword) fused with reciprocal rank fusion
    use_hybrid: bool = True
    fetch_k: int = 20
    # LLM reranker: scores each candidate 0-10, keeps the best top_k
    use_rerank: bool = True
    rerank_model: str = "gpt-4o-mini"
    min_rerank_score: int = 3

    admin_api_key: str = ""


@lru_cache
def get_settings() -> Settings:
    return Settings()
