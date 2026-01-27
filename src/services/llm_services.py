"""
LLM Service
LLM instance'larını yöneten servis
"""
from langchain_openai import ChatOpenAI
from src.core.config import get_settings

class LLMService:
    """LLM factory ve yönetim servisi"""
    
    def __init__(self):
        self.settings = get_settings()
    
    def get_llm(self, temperature: float = None) -> ChatOpenAI:
        """
        Standard LLM instance
        
        Args:
            temperature: Override default temperature
        """
        return ChatOpenAI(
            model=self.settings.LLM_MODEL,
            temperature=temperature or self.settings.LLM_TEMPERATURE,
            api_key=self.settings.OPENAI_API_KEY
        )
    
    def get_structured_llm(self, pydantic_model):
        """
        Structured output için LLM
        Router ve grader'larda kullanılır
        
        Args:
            pydantic_model: Pydantic model class
        """
        return self.get_llm().with_structured_output(pydantic_model)