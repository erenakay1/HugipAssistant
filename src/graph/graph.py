"""
Graph Builder
Tüm node'ları birleştirip graph'ı oluşturur
"""
from langgraph.graph import StateGraph, START, END
from src.graph.state import GraphState
from src.graph.nodes import (
    RouterNode,
    RetrieveNode,
    GenerateRAGNode,
    DirectNode,
    WebSearchNode
)
from src.graph.nodes.reflection import ReflectionNode


def build_graph():
    """
    Kulüp Asistanı Graph'ını oluşturur (Reflection ile)
    
    Flow:
    1. Router → Karar ver (rag/direct/web_search)
    2. RAG Path:
       - Retrieve → Dökümanları getir
       - Generate → Cevap üret
       - Reflection → Kalite kontrol (Hallucination check)
       - Eğer hallucination → Regenerate (max 2x)
       - Eğer grounded → Response
    3. Direct/Web Search → Direkt response
    
    Returns:
        Compiled LangGraph app
    """
    # Node instance'ları
    router_node = RouterNode()
    retrieve_node = RetrieveNode()
    generate_rag_node = GenerateRAGNode()
    reflection_node = ReflectionNode()
    direct_node = DirectNode()
    web_search_node = WebSearchNode()
    
    # Workflow oluştur
    workflow = StateGraph(GraphState)
    
    # Node'ları ekle
    workflow.add_node("router", router_node)
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("generate_rag", generate_rag_node)
    workflow.add_node("reflection", reflection_node)
    workflow.add_node("direct", direct_node)
    workflow.add_node("web_search", web_search_node)
    
    # Edge'leri ekle
    workflow.add_edge(START, "router")
    
    # Conditional routing
    def route_question(state: GraphState) -> str:
        """Router'dan sonra hangi node'a gidileceğini belirle"""
        decision = state["decision"]
        
        if decision == "rag":
            return "retrieve"
        elif decision == "web_search":
            return "web_search"
        else:
            return "direct"
    
    workflow.add_conditional_edges(
        "router",
        route_question,
        {
            "retrieve": "retrieve",
            "web_search": "web_search",
            "direct": "direct"
        }
    )
    
    # RAG flow (Reflection ile)
    workflow.add_edge("retrieve", "generate_rag")
    workflow.add_edge("generate_rag", "reflection")
    
    # Reflection sonrası conditional edge
    def check_reflection(state: GraphState) -> str:
        """
        Reflection sonrası karar:
        - Eğer max iteration aşıldı → END
        - Eğer hallucination check passed → END
        - Eğer hallucination detected ve iteration kaldı → Regenerate
        """
        iterations = state.get("iterations", 0)
        max_iterations = 3  # Router(0) + Generate(1) + Max 2 reflection
        
        # Max iteration aşıldı mı?
        if iterations >= max_iterations:
            return "end"
        
        # Son reflection sonucu neydi kontrol et
        # (ReflectionNode zaten regenerate etti, biz sadece flow'u yönetiyoruz)
        # Eğer iterations arttıysa demek ki ya grounded ya da regenerate edildi
        
        # Basit: Her zaman END'e git çünkü ReflectionNode içinde loop var
        return "end"
    
    workflow.add_conditional_edges(
        "reflection",
        check_reflection,
        {
            "end": END
        }
    )

    # Web search flow
    workflow.add_edge("web_search", END)
    
    # Direct flow
    workflow.add_edge("direct", END)
    
    # Compile
    return workflow.compile()