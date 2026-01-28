"""
Relevance Grader
Retrieved dökümanların soruyla alakalı olup olmadığını kontrol eder
"""
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from typing import List
from langchain_core.documents import Document
from src.services.llm_services import LLMService

class GradeRelevance(BaseModel):
    """
    Document relevance grading için structured output
    
    Attributes:
        binary_score: Döküman soruyla alakalı mı? (True/False)
        reasoning: Karar verme mantığı
    """
    binary_score: bool = Field(
        description="Döküman soruyla alakalı mı? True = relevant, False = not relevant"
    )
    reasoning: str = Field(
        description="Karar verme nedeni (kısa açıklama)"
    )

class RelevanceGrader:
    """
    Document relevance kontrolü yapan grader
    
    Retrieved dökümanların soruyla alakalı olup olmadığını kontrol eder.
    Alakasız dökümanları filtreleyerek generation kalitesini artırır.
    """
    
    def __init__(self):
        self.llm_service = LLMService()
        self.llm = self.llm_service.get_structured_llm(GradeRelevance)
        
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """Sen bir alakalılık denetleyicisisin (relevance checker).

Görevi: Retrieved dökümanın kullanıcı sorusuyla alakalı olup olmadığını değerlendir.

KURALLAR:
1. Döküman soruyla alakalı anahtar kelimeler içeriyor mu?
2. Döküman sorunun cevabına yardımcı olur mu?
3. Genel bilgi bile olsa, soruyla bağlantılı mı?

İlgili (True):
  - Sorudaki konu döküman içinde geçiyor
  - Döküman sorunun cevabına katkı sağlayabilir
  - Anlamsal olarak alakalı

İlgisiz (False):
  - Döküman tamamen farklı konu
  - Sorunun cevabına hiç katkı sağlamaz
  - Sadece benzer kelimeler var ama anlam farklı

ÖRNEKLER:

Soru: "FESTUP ne zaman?"
Döküman: "FESTUP 4 Aralık'ta 12:00-18:00 saatleri arasında..."
→ True (Doğrudan alakalı!)

Soru: "FESTUP ne zaman?"
Döküman: "DigitalMAG etkinliğinde 35+ marka katıldı..."
→ False (Farklı etkinlik, alakasız)

Soru: "Kulüp etkinlikleri neler?"
Döküman: "FESTUP, startup festivali konseptinde..."
→ True (Etkinlik bilgisi, alakalı)

Soru: "Social Media Talks'ta kimler var?"
Döküman: "Yönetim kurulu seçim süreci şöyledir..."
→ False (Farklı konu, alakasız)
"""),
            ("human", """Soru:
{question}

Döküman:
{document}

Bu döküman soruyla alakalı mı?""")
        ])
    
    def grade(
        self, 
        question: str, 
        document: Document
    ) -> GradeRelevance:
        """
        Tek bir dökümanın relevance kontrolü
        
        Args:
            question: Kullanıcı sorusu
            document: Retrieved döküman
            
        Returns:
            GradeRelevance: binary_score (True/False) ve reasoning
        """
        chain = self.prompt | self.llm
        result = chain.invoke({
            "question": question,
            "document": document.page_content
        })
        
        return result
    
    def filter_documents(
        self, 
        question: str, 
        documents: List[Document]
    ) -> List[Document]:
        """
        Döküman listesini filtrele, sadece relevant olanları döndür
        
        Args:
            question: Kullanıcı sorusu
            documents: Retrieved dökümanlar
            
        Returns:
            Filtered list of relevant documents
        """
        filtered_docs = []
        
        for doc in documents:
            result = self.grade(question, doc)
            if result.binary_score:
                filtered_docs.append(doc)
        
        return filtered_docs
    
    def grade_all(
        self, 
        question: str, 
        documents: List[Document]
    ) -> List[tuple[Document, bool, str]]:
        """
        Tüm dökümanları grade et ve sonuçları döndür
        (Debugging için kullanışlı)
        
        Returns:
            List of (document, is_relevant, reasoning)
        """
        results = []
        
        for doc in documents:
            grade = self.grade(question, doc)
            results.append((doc, grade.binary_score, grade.reasoning))
        
        return results