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

def build_graph():
    """
    Kulüp Asistanı Graph'ını oluşturur
    
    Returns:
        Compiled LangGraph app
    """
    # Node instance'ları
    router_node = RouterNode()
    retrieve_node = RetrieveNode()
    generate_rag_node = GenerateRAGNode()
    direct_node = DirectNode()
    web_search_node = WebSearchNode()
    
    # Workflow oluştur
    workflow = StateGraph(GraphState)
    
    # Node'ları ekle
    workflow.add_node("router", router_node)
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("generate_rag", generate_rag_node)
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
    
    # RAG flow
    workflow.add_edge("retrieve", "generate_rag")
    workflow.add_edge("generate_rag", END)
    
    # Web search flow
    workflow.add_edge("web_search", END)
    
    # Direct flow
    workflow.add_edge("direct", END)
    
    # Compile
    return workflow.compile()