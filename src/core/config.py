"""
Core Configuration
Environment variables ve settings
"""
from pydantic_settings import BaseSettings
from functools import lru_cache

class Settings(BaseSettings):
    """Uygulama ayarları"""
    
    # API Ayarları
    APP_NAME: str = "Kulüp Asistanı AI Service"
    VERSION: str = "1.0.0"
    API_PREFIX: str = "/api/v1"
    
    # LangSmith (Observability)
    LANGCHAIN_TRACING_V2: bool = True
    LANGCHAIN_ENDPOINT: str = "https://api.smith.langchain.com"
    LANGCHAIN_API_KEY: str
    LANGCHAIN_PROJECT: str = "Hugip Assistant"
    
    # LLM Ayarları
    OPENAI_API_KEY: str
    LLM_MODEL: str = "gpt-4o-mini"
    LLM_TEMPERATURE: float = 0.0
    
    # Pinecone
    PINECONE_API_KEY: str
    PINECONE_INDEX_NAME: str = "hugip-doc-index"
    
    # Tavily Search (Opsiyonel)
    TAVILY_API_KEY: str = ""
    
    # RAG Settings
    RETRIEVAL_K: int = 4
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200
    
    # .NET Backend Integration (İleride kullanılacak)
    DOTNET_BACKEND_URL: str = "http://localhost:5000"
    DOTNET_API_KEY: str = ""
    
    class Config:
        env_file = ".env"
        case_sensitive = True
        extra = "ignore"  # Extra env variables'ları yoksay

@lru_cache()
def get_settings() -> Settings:
    """Singleton settings instance"""
    return Settings()