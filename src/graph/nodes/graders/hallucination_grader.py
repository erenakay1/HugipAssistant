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
            4. Cevap belgelere sadıksa → True

            ÖRNEKLER:

            Belgeler: "FESTUP 4 Aralık'ta yapılacak"
            Cevap: "FESTUP 4 Aralık'ta yapılacak"
            → True (Doğru bilgi)

            Belgeler: "FESTUP 4 Aralık'ta yapılacak"
            Cevap: "FESTUP 5 Aralık'ta yapılacak"
            → False (TARİH YANLIŞ! Hallucination)

            Belgeler: "Konuşmacılar: Melih Abuaf, Sinan Koç"
            Cevap: "Konuşmacılar: Melih Abuaf, Ahmet Yılmaz, Sinan Koç"
            → False (Ahmet Yılmaz belgede YOK! Hallucination)

            Belgeler: "Kulüp 2020'de kuruldu"
            Cevap: "Kulüp yıllardır aktif"
            → True (Genel ifade, belgelerle çelişmiyor)
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
        # Documents'ları string'e çevir
        docs_content = "\n\n---\n\n".join([
            f"[Döküman {i+1}]\n{doc.page_content}"
            for i, doc in enumerate(documents)
        ])
        
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