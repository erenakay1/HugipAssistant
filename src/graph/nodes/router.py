"""
Router Node
Soruyu analiz edip uygun datasource'a yönlendirir
MEMORY: Conversation history kullanır
"""
from typing import Literal
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from src.graph.state import GraphState
from src.services.llm_services import LLMService
from src.services.memory_service import MemoryService

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
        self.memory_service = MemoryService()
        self.llm = self.llm_service.get_structured_llm(RouteDecision)
        
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """Sen Haliç Üniversitesi Girişimcilik ve Pazarlama Kulübü asistanısın.

Soruyu analiz et ve uygun veri kaynağına yönlendir:

- 'rag': Kulüp hakkında sorular
  * KULÜP BİLGİLERİ: Amaç, vizyon, misyon, kuruluş, organizasyon
  * Tüzük, yönetim, üyelik, kurallar
  * Ekipler ve görev tanımları (Dış İlişkiler, Operasyon, vb.)
  * ETKİNLİKLER: DigitalMAG, FESTUP, HUGİP Akademi, Social Media Talks
  * Etkinlik konuşmacıları, tarih/saat bilgileri
  * İş/staj fırsatları, networking, katılım şartları
  * İletişim bilgileri
  
- 'web_search': Güncel haberler, genel bilgiler, kulüp dışı konular
  * Güncel haberler (bugünkü hava, son dakika)
  * Genel tanımlar (yapay zeka nedir, blockchain nedir)
  * Spor sonuçları, borsa bilgileri vb.
  
- 'direct': Basit selamlamalar, teşekkürler, genel sohbet

ÖNEMLİ: 
- "Kulüp", "etkinlik", "üye", "katılım", "amaç", "vizyon" → 'rag'
- Etkinlik isimleri (DigitalMAG, FESTUP, Social Media Talks, HUGİP Akademi) → 'rag'
- "Ne zaman?", "Kimler?", "Nerede?" gibi takip soruları → Önceki context'e bak!
  * Eğer önceki konuşmada etkinlik/kulüp konuşuluyorsa → 'rag'
  * Eğer önceki konuşmada genel bir konu varsa → 'web_search'

ÖRNEKLER:
✓ "Kulübün amacı nedir?" → rag
✓ "FESTUP nedir?" → rag
✓ [Önceki: "FESTUP nedir?"] "Ne zaman yapılıyor?" → rag (FESTUP context'i)
✓ [Önceki: "Social Media Talks"] "Kimler konuşacak?" → rag (Etkinlik context'i)
✓ "Merhaba!" → direct
"""),
            ("human", """CONVERSATION HISTORY (Son mesajlar):
{history}

ŞİMDİKİ SORU:
{question}

Bu soruyu hangi datasource'a yönlendirmeliyim?""")
        ])
    
    def __call__(self, state: GraphState) -> GraphState:
        """
        Router logic with conversation memory
        
        Args:
            state: Mevcut graph state
            
        Returns:
            Updated state with decision
        """
        session_id = state.get("session_id", "default")
        
        # User mesajını memory'ye kaydet (ilk kez)
        self.memory_service.add_user_message(session_id, state["question"])
        
        # Conversation history'yi al (son 6 mesaj = 3 turn)
        history = self.memory_service.get_context(session_id, last_n=6)

        last_topic = self.memory_service.get_last_topic(session_id)
        
        # History yoksa boş string
        if not history:
            history = "(İlk mesaj, önceki konuşma yok)"
        
        # Topic varsa history'ye ekle
        if last_topic:
            history = f"[DEVAM EDEN KONU: {last_topic}]\n\n{history}"

        chain = self.prompt | self.llm
        decision = chain.invoke({
            "question": state["question"],
            "history": history
        })
        
        return {
            **state,
            "decision": decision.datasource
        }