"""
Router Node
Soruyu analiz edip uygun datasource'a yönlendirir
"""
from typing import Literal
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from src.graph.state import GraphState
from src.services.llm_services import LLMService

class RouteDecision(BaseModel):
    """Router kararı için structured output"""
    datasource: Literal["rag", "web_search", "direct"] = Field(
        description="Veri kaynağı seçimi"
    )
    reasoning: str = Field(
        description="Karar verme mantığı"
    )

class RouterNode:
    """Router node class"""
    
    def __init__(self):
        self.llm_service = LLMService()
        self.llm = self.llm_service.get_structured_llm(RouteDecision)
        
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """Sen Haliç Üniversitesi Girişimcilik ve Pazarlama Kulübü asistanısın.

Soruyu analiz et ve uygun veri kaynağına yönlendir:

- 'rag': Kulüp hakkında sorular
  * Tüzük, yönetim, üyelik, organizasyon
  * Ekipler ve görev tanımları (Dış İlişkiler, Operasyon, vb.)
  * ETKİNLİKLER: DigitalMAG, FESTUP, HUGİP Akademi, Social Media Talks
  * Etkinlik konuşmacıları, tarih/saat bilgileri
  * İş/staj fırsatları, networking, katılım şartları
  
- 'web_search': Güncel haberler, genel bilgiler, kulüp dışı konular
  * Güncel haberler
  * Genel tanımlar (yapay zeka nedir, blockchain nedir)
  * Hava durumu, spor sonuçları vb.
  
- 'direct': Basit selamlamalar, teşekkürler, genel sohbet

ÖNEMLİ: Etkinlik isimleri (DigitalMAG, FESTUP, Social Media Talks, HUGİP Akademi) 
veya "etkinlik", "konuşmacı", "katılım", "staj/iş fırsatı" gibi kelimeler varsa → 'rag'

ÖRNEKLER:
✓ "Kulübe nasıl üye olurum?" → rag
✓ "Yönetim kurulu kimlerden oluşur?" → rag
✓ "Dış İlişkiler ekibi ne yapar?" → rag
✓ "FESTUP'ta iş bulabilir miyim?" → rag (FESTUP kulüp etkinliği!)
✓ "Social Media Talks ne zaman?" → rag (Kulüp etkinliği!)
✓ "Melih Aktaş kimdir?" → rag (Etkinlik konuşmacısı!)
✓ "DigitalMAG'a kaç kişi katılıyor?" → rag (Kulüp etkinliği!)
✓ "Yapay zeka nedir?" → web_search (Genel bilgi)
✓ "Bugün hava nasıl?" → web_search (Güncel bilgi)
✓ "Merhaba!" → direct
✓ "Teşekkürler!" → direct
"""),
            ("human", "{question}")
        ])
    
    def __call__(self, state: GraphState) -> GraphState:
        """
        Router logic
        
        Args:
            state: Mevcut graph state
            
        Returns:
            Updated state with decision
        """
        chain = self.prompt | self.llm
        decision = chain.invoke({"question": state["question"]})
        
        return {
            **state,
            "decision": decision.datasource
        }