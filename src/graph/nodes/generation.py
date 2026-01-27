"""
Direct ve Web Search Nodes
"""
from langchain_core.prompts import ChatPromptTemplate
from src.graph.state import GraphState
from src.services.llm_services import LLMService

class DirectNode:
    """Direkt cevap node'u (selamlama, genel sohbet)"""
    
    def __init__(self):
        self.llm_service = LLMService()
        self.llm = self.llm_service.get_llm()
        
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """Sen Haliç Üniversitesi Girişimcilik ve Pazarlama Kulübü asistanısın.

Samimi ve yardımsever bir şekilde cevap ver.
Kulüp hakkında soru sorulursa kullanıcıyı yönlendir.
Kısa ve doğal konuş.
"""),
            ("human", "{question}")
        ])
    
    def __call__(self, state: GraphState) -> GraphState:
        """
        Direct response generation
        
        Args:
            state: Mevcut graph state
            
        Returns:
            Updated state with generation
        """
        chain = self.prompt | self.llm
        response = chain.invoke({"question": state["question"]})
        
        return {
            **state,
            "generation": response.content
        }

class WebSearchNode:
    """Web search node (Tavily API)"""
    
    def __init__(self):
        self.llm_service = LLMService()
        self.llm = self.llm_service.get_llm()
        # TODO: Tavily client eklenecek
    
    def __call__(self, state: GraphState) -> GraphState:
        """
        Web search (şimdilik simulated)
        
        Args:
            state: Mevcut graph state
            
        Returns:
            Updated state with generation
        """
        # TODO: Gerçek Tavily entegrasyonu
        generation = (
            "Web araması henüz aktif değil. Bu özellik için Tavily API key "
            "eklemeniz gerekiyor. Kulüp hakkında sorularınız için dökümanlarımızı "
            "kullanabilirsiniz."
        )
        
        return {
            **state,
            "generation": generation,
            "web_results": ["Web search not configured"]
        }