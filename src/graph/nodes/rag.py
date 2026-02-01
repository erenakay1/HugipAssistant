"""
RAG Nodes
Retrieve ve Generate node'ları
"""
from langchain_core.prompts import ChatPromptTemplate
from src.graph.state import GraphState
from src.services.vectorstore_service import VectorStoreService
from src.services.llm_services import LLMService

class RetrieveNode:
    """Pinecone'dan döküman getiren node"""
    
    def __init__(self):
        self.vectorstore_service = VectorStoreService()
    
    def __call__(self, state: GraphState) -> GraphState:
        """
        Pinecone'dan döküman retrieval
        
        Args:
            state: Mevcut graph state
            
        Returns:
            Updated state with documents
        """
        documents = self.vectorstore_service.retrieve_documents(
            query=state["question"]
        )
        
        return {
            **state,
            "documents": documents
        }

class GenerateRAGNode:
    """RAG ile cevap üreten node"""
    
    def __init__(self):
        self.llm_service = LLMService()
        self.llm = self.llm_service.get_llm()
        
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """Sen Haliç Üniversitesi Girişimcilik ve Pazarlama Kulübü asistanısın.

CONTEXT'teki dökümanları kullanarak soruyu yanıtla.

KURALLAR:
1. Sadece CONTEXT'teki bilgileri kullan
2. ETKİNLİK sorularında MUTLAKA spesifik isimlerini belirt (FESTUP, Social Media Talks, DigitalMAG, HUGİP Akademi vb.)
3. "Etkinlikler", "festivaller", "konferanslar" gibi genel ifadeler YERİNE구체 isimleri say
4. Örnek: "FESTUP, Social Media Talks ve DigitalMAG gibi etkinlikler düzenliyoruz"
5. Eğer bilgi yoksa "Bu konuda dökümanlarımda detaylı bilgi bulamadım" de
6. Kısa, öz ve samimi ol
7. Kaynak belirtmeye gerek yok (otomatik gösterilecek)
8. Türkçe sorulara Türkçe, İngilizce sorulara İngilizce cevap ver

ÖNEMLİ: Context'te etkinlik/proje isimleri görüyorsan, bunları cevabında MUTLAKA kullan!

CONTEXT:
{context}
"""),
            ("human", "{question}")
        ])
    
    def __call__(self, state: GraphState) -> GraphState:
        """
        RAG generation
        
        Args:
            state: Mevcut graph state (documents ile)
            
        Returns:
            Updated state with generation
        """
        # Context hazırla
        context = "\n\n---\n\n".join([
            f"[Kaynak: {doc.metadata.get('source', 'Unknown')}]\n{doc.page_content}"
            for doc in state["documents"]
        ])
        
        # Generate
        chain = self.prompt | self.llm
        response = chain.invoke({
            "context": context,
            "question": state["question"]
        })
        
        return {
            **state,
            "generation": response.content
        }