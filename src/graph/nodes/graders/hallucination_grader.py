"""
Hallucination Grader
LLM'in ürettiği cevabın documents'a sadık olup olmadığını kontrol eder
"""
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from typing import List
from langchain_core.documents import Document
from src.services.llm_services import LLMService

class GradeHallucination(BaseModel):
    """
    Hallucination grading için structured output
    
    Attributes:
        binary_score: Cevap documents'a sadık mı? (True/False)
        reasoning: Karar verme mantığı
    """
    binary_score: bool = Field(
        description="Cevap belgelerden üretilmiş mi? True = sadık, False = hallucination"
    )
    reasoning: str = Field(
        description="Karar verme nedeni (kısa açıklama)"
    )

class HallucinationGrader:
    """
    Hallucination kontrolü yapan grader
    
    LLM'in ürettiği cevabın, verilen dökümanlardan üretilip üretilmediğini
    veya uydurma bilgi içerip içermediğini kontrol eder.
    """
    
    def __init__(self):
        self.llm_service = LLMService()
        self.llm = self.llm_service.get_structured_llm(GradeHallucination)
        
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """Sen bir doğruluk denetleyicisisin (fact-checker).

Görevi: LLM tarafından üretilen cevabın, verilen BELGELER'e (documents) sadık olup 
olmadığını kontrol et.

KURALLAR:
1. Cevaptaki HER BİLGİ belgelerden gelmeli
2. Tarih, saat, isim gibi SPESİFİK BİLGİLER tam olarak eşleşmeli
3. Belgede OLMAYAN bilgi varsa → Hallucination! (False)
4. Eğer cevap KISMEN DOĞRU ve eksik bilgi varsa:
   - Belgede var ama cevap kısa ise → True (eksik ama doğru)
   - Cevap uydurmuyorsa sadece kısa ise → True
5. Cevap belgelere sadıksa → True

**ÇOK ÖNEMLİ - "BİLGİ YOK" CEVAPLARI:**
- Eğer belgeler BİLGİ İÇERİYOR ama cevap "bilgi bulamadım" diyorsa → FALSE (Hallucination!)
- Örnek: Belgede "FESTUP, Social Media Talks" var, cevap "bilgi yok" diyor → FALSE
- Bu tür cevaplar YANLIŞ çünkü belgede bilgi VAR ama LLM kullanmıyor!

**ÖNEMLİ - ETKİNLİK/PROJE İSİMLERİ:**
- Belgede etkinlik isimleri geçiyorsa (FESTUP, DigitalMAG vb.), cevap bunları belirtmelidir
- Eğer belgede "FESTUP" var ama cevap belirtmiyorsa → Eksik ama True (kısmen cevap)
- Eğer belgede "FESTUP" YOK ama cevap "FESTUP var" diyorsa → FALSE (Hallucination!)
- Eğer belgede birden fazla etkinlik var, cevap sadece bazılarını sayıyorsa → True (eksik ama doğru)

ÖNEMLİ: 
- Liste soruları (kimler, neler) için: Cevaptaki TÜM İSİMLER belgede olmalı
- Eğer bazı isimler belgede VAR, bazıları YOK → False
- Eğer tüm isimler belgede var ama liste eksik → True (kısmen cevap normal)

ÖRNEKLER:

Belgeler: "FESTUP 4 Aralık'ta yapılacak"
Cevap: "FESTUP 4 Aralık'ta yapılacak"
→ True (Doğru bilgi)

Belgeler: "FESTUP 4 Aralık'ta yapılacak"
Cevap: "FESTUP 5 Aralık'ta yapılacak"
→ False (TARİH YANLIŞ! Hallucination)

Belgeler: "FESTUP, Social Media Talks, DigitalMAG"
Cevap: "Bu konuda detaylı bilgi bulamadım"
→ False (Belgede BİLGİ VAR ama kullanmamış! Hallucination)

Belgeler: "Konuşmacılar: Melih Abuaf, Sinan Koç, Ahmet Yılmaz"
Cevap: "Konuşmacılar: Melih Abuaf, Sinan Koç"
→ True (İKİSİ DE BELGEDE VAR, eksik ama doğru)

Belgeler: "Konuşmacılar: Melih Abuaf, Sinan Koç"
Cevap: "Konuşmacılar: Melih Abuaf, Ahmet Yılmaz, Sinan Koç"
→ False (Ahmet Yılmaz belgede YOK! Hallucination)

Belgeler: "Kulüp 2020'de kuruldu"
Cevap: "Kulüp yıllardır aktif"
→ True (Genel ifade, belgelerle çelişmiyor)

Belgeler: [15 konuşmacı listesi... Sinan Koç en sonda]
Cevap: "İlk konuşmacılar: 1453 Harun, Goktug Alaf, Sinan Koç"
→ True (HER ÜÇ İSİM DE BELGEDE VAR! Eksik ama doğru)
"""),
            ("human", """Belgeler:
{documents}

LLM Cevabı:
{generation}

Bu cevap belgelere sadık mı, yoksa uydurma bilgi içeriyor mu?""")
        ])
    
    def grade(
        self, 
        generation: str, 
        documents: List[Document]
    ) -> GradeHallucination:
        """
        Hallucination kontrolü yap
        
        Args:
            generation: LLM'in ürettiği cevap
            documents: Retrieved dökümanlar
            
        Returns:
            GradeHallucination: binary_score (True/False) ve reasoning
        """
        # Documents'ları string'e çevir (KAYNAK İSİMLERİYLE)
        docs_content = "\n\n---\n\n".join([
            f"[Döküman {i+1} - Kaynak: {doc.metadata.get('source', 'Unknown')}]\n{doc.page_content[:1500]}"  # İlk 1500 karakter
            for i, doc in enumerate(documents)
        ])
        
        # Debugging: Döküman kaynaklarını log'la
        print(f"   📄 Grading with {len(documents)} documents:")
        for i, doc in enumerate(documents):
            source = doc.metadata.get('source', 'Unknown')
            content_preview = doc.page_content[:100].replace('\n', ' ')
            print(f"      {i+1}. {source}: {content_preview}...")
        
        # Grade
        chain = self.prompt | self.llm
        result = chain.invoke({
            "documents": docs_content,
            "generation": generation
        })
        
        return result
    
    def is_grounded(
        self, 
        generation: str, 
        documents: List[Document]
    ) -> bool:
        """
        Basit bool döndüren helper method
        
        Returns:
            True: Cevap documents'a sadık
            False: Hallucination var
        """
        result = self.grade(generation, documents)
        return result.binary_score