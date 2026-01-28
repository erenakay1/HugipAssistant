"""
Reflection Node
Generation'ın kalitesini kontrol eder ve gerekirse regenerate eder
"""
from src.graph.state import GraphState
from src.graph.nodes.graders import HallucinationGrader
from src.services.llm_services import LLMService
from langchain_core.prompts import ChatPromptTemplate

class ReflectionNode:
    """
    Level 1 Reflection: Sadece Hallucination Check
    
    Generation'ı kontrol eder:
    - Hallucination var mı? → Regenerate
    - Hallucination yok mu? → Approve
    """
    
    def __init__(self):
        self.hallucination_grader = HallucinationGrader()
        self.llm_service = LLMService()
        self.llm = self.llm_service.get_llm()
        
        # Regeneration prompt (daha dikkatli)
        self.regenerate_prompt = ChatPromptTemplate.from_messages([
            ("system", """Sen Haliç Üniversitesi Girişimcilik ve Pazarlama Kulübü asistanısın.

            ÖNCEKİ CEVABINDA HATA VARMIŞ! Lütfen daha DİKKATLİ ol.

            KURALLAR:
            1. SADECE verilen CONTEXT'teki bilgileri kullan
            2. Tarih, saat, isim gibi bilgileri TAM OLARAK yaz
            3. CONTEXT'te OLMAYAN bilgi verme
            4. Emin değilsen "Bu konuda detaylı bilgi bulamadım" de

            CONTEXT:
            {context}
            """),
                        ("human", "{question}")
                    ])
    
    def __call__(self, state: GraphState) -> GraphState:
        """
        Reflection logic
        
        Args:
            state: Mevcut graph state (generation ile)
            
        Returns:
            Updated state (onaylanmış veya regenerate edilmiş)
        """
        question = state["question"]
        generation = state["generation"]
        documents = state["documents"]
        iterations = state.get("iterations", 0)
        max_iterations = 2  # Max 2 regeneration
        
        print(f"\n   🔍 Reflection Node: Checking quality (iteration {iterations})...")
        
        # Hallucination check
        hallucination_result = self.hallucination_grader.grade(
            generation=generation,
            documents=documents
        )
        
        print(f"   📊 Hallucination Check: {hallucination_result.binary_score}")
        print(f"   💭 Reasoning: {hallucination_result.reasoning}")
        
        # Eğer grounded ise (hallucination yok)
        if hallucination_result.binary_score:
            print("   ✅ Quality check PASSED! Cevap documents'a sadık.")
            return {
                **state,
                "iterations": iterations + 1
            }
        
        # Hallucination var!
        print("   ❌ Quality check FAILED! Hallucination detected!")
        
        # Max iteration aşıldı mı?
        if iterations >= max_iterations:
            print(f"   ⚠️  Max iteration ({max_iterations}) reached. Using best attempt.")
            return {
                **state,
                "generation": generation + "\n\n(Not: Bu bilgi dökümanlarımızda tam olarak bulunamadı. Lütfen kulüple direkt iletişime geçin.)",
                "iterations": iterations + 1
            }
        
        # Regenerate
        print("   🔄 Regenerating with more careful prompt...")
        
        # Context hazırla
        context = "\n\n---\n\n".join([
            f"[Kaynak: {doc.metadata.get('source', 'Unknown')}]\n{doc.page_content}"
            for doc in documents
        ])
        
        # Regenerate
        chain = self.regenerate_prompt | self.llm
        new_response = chain.invoke({
            "context": context,
            "question": question
        })
        
        print(f"   ✅ Regenerated ({len(new_response.content)} characters)")
        
        return {
            **state,
            "generation": new_response.content,
            "iterations": iterations + 1
        }