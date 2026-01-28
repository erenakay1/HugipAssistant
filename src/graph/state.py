"""
Graph State
LangGraph state tanımı
"""
from typing import TypedDict, List
from langchain_core.documents import Document

class GraphState(TypedDict):
    """
    Kulüp Asistanı Graph State
    
    Tüm graph boyunca paylaşılan state yapısı
    
    Attributes:
        question: Kullanıcının orijinal sorusu
        generation: LLM tarafından üretilen cevap
        documents: Retrieved dökümanlar (RAG)
        decision: Router kararı ('rag', 'web_search', 'direct')
        web_results: Web arama sonuçları (opsiyonel)
        iterations: Reflection loop sayacı
        session_id: Conversation session identifier
    """
    question: str
    generation: str
    documents: List[Document]
    decision: str
    web_results: List[str]
    iterations: int
    session_id: str