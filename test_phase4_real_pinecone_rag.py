"""
PHASE 4: Gerçek Pinecone ile RAG Test
Mock yerine gerçek Pinecone kullanıyoruz!
"""
import os
from dotenv import load_dotenv
from typing import TypedDict, List
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langgraph.graph import StateGraph, START, END

load_dotenv()

print("=" * 60)
print("🧪 PHASE 4: Gerçek Pinecone ile RAG Testi")
print("=" * 60)

# Config
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "hugip-doc-index")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")

# State
class GraphState(TypedDict):
    question: str
    documents: List[Document]
    generation: str

# Gerçek Pinecone Vectorstore
print("\n1️⃣ Pinecone'a bağlanılıyor...")
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    openai_api_key=os.getenv("OPENAI_API_KEY")
)

vectorstore = PineconeVectorStore(
    index_name=PINECONE_INDEX_NAME,
    embedding=embeddings
)
print(f"   ✅ Pinecone bağlantısı başarılı! (Index: {PINECONE_INDEX_NAME})")

# LLM
llm = ChatOpenAI(model=LLM_MODEL, temperature=0.0)

# RAG Node
def retrieve_node(state: GraphState) -> GraphState:
    """Gerçek Pinecone'dan döküman getir"""
    print("\n   📚 RAG Node: Retrieving from Pinecone...")
    
    question = state["question"]
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    documents = retriever.invoke(question)
    
    print(f"   ✅ {len(documents)} döküman bulundu:")
    for i, doc in enumerate(documents, 1):
        source = doc.metadata.get('source', 'Unknown')
        # Windows path'i daha okunaklı yap
        source = source.split('\\')[-1] if '\\' in source else source
        print(f"      {i}. {source}")
    
    return {
        **state,
        "documents": documents
    }

# Generation Node
def generate_node(state: GraphState) -> GraphState:
    """RAG ile cevap üret"""
    print("\n   ✨ Generation Node: Creating answer...")
    
    # Context hazırla
    context = "\n\n".join([doc.page_content for doc in state["documents"]])
    
    # Prompt
    prompt = ChatPromptTemplate.from_messages([
        ("system", """Sen Haliç Üniversitesi Girişimcilik ve Pazarlama Kulübü asistanısın.

Verilen CONTEXT bilgilerini kullanarak soruyu Türkçe olarak yanıtla.

KURALLAR:
- Context'teki bilgilere sadık kal
- Eğer context'te cevap yoksa, "Bu konuda dökümanlarımda bilgi bulamadım" de
- Kısa ve öz cevaplar ver
- Samimi ve yardımsever ol

CONTEXT:
{context}
"""),
        ("human", "{question}")
    ])
    
    chain = prompt | llm
    response = chain.invoke({
        "context": context,
        "question": state["question"]
    })
    
    print(f"   ✅ Cevap üretildi ({len(response.content)} karakter)")
    
    return {
        **state,
        "generation": response.content
    }

# Graph oluştur
print("\n2️⃣ RAG Graph oluşturuluyor...")

workflow = StateGraph(GraphState)
workflow.add_node("retrieve", retrieve_node)
workflow.add_node("generate", generate_node)

workflow.add_edge(START, "retrieve")
workflow.add_edge("retrieve", "generate")
workflow.add_edge("generate", END)

app = workflow.compile()

print("   ✅ Graph başarıyla oluşturuldu!")

# Test Soruları
test_questions = [
    "Kulübe nasıl üye olabilirim?",
    "Kulübün amacı nedir?",
    "Yönetim Kurulu nasıl seçilir?",
    "Kulüp hangi etkinlikler düzenler?",
]

print("\n" + "=" * 60)
print("3️⃣ Gerçek RAG Testi Başlıyor:\n")

for i, question in enumerate(test_questions, 1):
    print(f"\n{'='*60}")
    print(f"Soru {i}: '{question}'")
    print("=" * 60)
    
    # Initial state
    initial_state = {
        "question": question,
        "documents": [],
        "generation": ""
    }
    
    # Run graph
    result = app.invoke(initial_state)
    
    print("\n📝 CEVAP:")
    print("-" * 60)
    print(result['generation'])
    print("-" * 60)
    
    print(f"\n📚 Kaynak dökümanlar:")
    for j, doc in enumerate(result['documents'], 1):
        source = doc.metadata.get('source', 'Unknown')
        source = source.split('\\')[-1] if '\\' in source else source
        print(f"   {j}. {source}")

print("\n" + "=" * 60)
print("✅ PHASE 4 TAMAMLANDI!")
print("=" * 60)
print("\n💡 Başardıkların:")
print("   ✅ Gerçek Pinecone ile RAG çalışıyor!")
print("   ✅ PDF'lerden bilgi getiriliyor")
print("   ✅ LLM context'e göre cevap üretiyor")
print("   ✅ LangSmith'te her şey trace'leniyor")
print("\n🔗 LangSmith'te trace'leri incele:")
print("   - Her sorunun retrieval sonuçlarını gör")
print("   - Generation prompt'unu incele")
print("   - Context'in nasıl kullanıldığını analiz et")
print(f"   https://smith.langchain.com/o/YOUR_ORG/projects/p/{os.getenv('LANGCHAIN_PROJECT', 'club-assistant-dev')}")