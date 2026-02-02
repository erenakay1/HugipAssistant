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
  
- 'web_search': SADECE HUGİP ile bağlantılı ama güncel bilgi gerekli olan sorular
  * "HUGİP'in son etkinliği ne zaman?" (güncel bilgi)
  * "HUGİP Instagram'ı ne zaman güncellendi?" (güncel)
  
- 'direct': Selamlamalar, teşekkürler VE HUGİP DIŞI KONULAR
  * "Merhaba", "Selam", "Teşekkürler"
  * Oyun oynamak, rol yapma
  * HUGİP ile ilgisi olmayan sorular:
    - "Türkiye'nin nüfusu nedir?"
    - "Instagram'da follower kazanmak"
    - "Python script yaz"
    - Genel bilgi, homework, coding vb.

ÖNEMLİ: 
- Eğer soru bir TASK/ACTION ise → HER ZAMAN 'direct'
  Task kelimeleri: yaz, oluştur, tasarla, code, script, yap, hazırla, düzenle, güncelle, değiştir, oluşturalım, hazırlayalım, yeniden yaz, yapayım, yapayalım
  * "HUGİP için Python script yaz" → direct (task!)
  * "HUGİP logo tasarla" → direct (task!)
  * "Etkinlik için poster yaz" → direct (task!)
  * "Bir etkinlik takvimi hazırlayalım" → direct (task! "hazırlayalım")
  * Yani yukarıdaki task kelimelerinden HERHANGI BİRİ varsa → direct
- Eğer soru BİLGI SORISI ise (nedir, kim, ne zaman, nasıl, nerede) → 'rag'
  * "HUGİP nedir?" → rag (bilgi sorusu)
  * "FESTUP ne zaman?" → rag (bilgi sorusu)
- "Kulüp", "etkinlik", "üye" kelimesi olsa bile task ise → direct
- Etkinlik isimleri (DigitalMAG, FESTUP, Social Media Talks, HUGİP Akademi) + bilgi sorusu → 'rag'
- "Ne zaman?", "Kimler?", "Nerede?" gibi takip soruları → Önceki context'e bak!
  * Eğer önceki konuşmada etkinlik/kulüp konuşuluyorsa → 'rag'
  * Eğer önceki konuşmada genel bir konu varsa → 'web_search'

ÖRNEKLER:
✓ "Kulübün amacı nedir?" → rag (bilgi sorusu)
✓ "FESTUP nedir?" → rag (bilgi sorusu)
✓ [Önceki: "FESTUP nedir?"] "Ne zaman yapılıyor?" → rag (FESTUP context'i)
✓ [Önceki: "Social Media Talks"] "Kimler konuşacak?" → rag (Etkinlik context'i)
✓ "Merhaba!" → direct (selam)
✓ "Türkiye'nin nüfusu nedir?" → direct (HUGİP dışı)
✓ "Instagram'da follower kazanmak?" → direct (HUGİP dışı)
✓ "Oyun oynayalım mı?" → direct (HUGİP dışı)
✓ "HUGİP için Python script yaz" → direct (TASK! "yaz" kelimesi var)
✓ "Etkinlik için poster tasarla" → direct (TASK! "tasarla" kelimesi var)
✓ "HUGİP logo oluştur" → direct (TASK! "oluştur" kelimesi var)
✓ "Bir etkinlik takvimi hazırlayalım" → direct (TASK! "hazırlayalım")
✓ "Yenisini hazırlayalım mı" → direct (TASK! "hazırlayalım")
✓ "HUGİP'in son etkinlik tarihi?" → web_search (HUGİP + güncel)
"""),
            ("human", """CONVERSATION HISTORY (Son mesajlar):
{history}

ŞİMDİKİ SORU:
{question}

ÖNEMLİ NOTLAR:
- Eğer history'de bir ETKİNLİK (FESTUP, Social Media Talks, vb.) veya KULÜP konusu varsa,
  takip soruları ("ne zaman?", "kimler?", "kaç kişi?", "iş var mı?") o konuyla ilgilidir!
- Örnek: Önceki mesajlarda "FESTUP" veya "etkinlik" geçiyorsa, 
  "İş bulabilir miyim?" = "FESTUP'ta iş bulabilir miyim?" demektir → 'rag'

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
        
        # User mesajını memory'ye kaydet
        # Bu şekilde her yerde (test, API) aynı davranış olur
        self.memory_service.add_user_message(session_id, state["question"])
        
        # Conversation history'yi al (son 6 mesaj = 3 turn)
        history = self.memory_service.get_context(session_id, last_n=6)
        
        # Son topic'i al (FESTUP, Social Media Talks vb.)
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