"""
Pinecone Retrieval Test
Yükleme sonrası retrieval'ı test et
"""
import os
from dotenv import load_dotenv
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings

load_dotenv()

print("=" * 60)
print("🔍 PINECONE RETRIEVAL TEST")
print("=" * 60)

# Config
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "hugip-doc-index")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not PINECONE_API_KEY or not OPENAI_API_KEY:
    print("❌ API keys eksik! .env dosyasını kontrol et")
    exit(1)

# 1. Pinecone Stats
print("\n1️⃣ Index İstatistikleri:")
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX_NAME)
stats = index.describe_index_stats()

print(f"   📊 Index: {PINECONE_INDEX_NAME}")
print(f"   📝 Total Records: {stats.total_vector_count}")
print(f"   💾 Dimension: {stats.dimension}")

if stats.total_vector_count == 0:
    print("\n   ⚠️  Index boş! Önce setup_pinecone.py çalıştır")
    exit(0)

# 2. Vectorstore Bağlantısı
print("\n2️⃣ Vectorstore'a bağlanılıyor...")
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    openai_api_key=OPENAI_API_KEY
)

vectorstore = PineconeVectorStore(
    index_name=PINECONE_INDEX_NAME,
    embedding=embeddings
)
print("   ✅ Bağlantı başarılı!")

# 3. Test Queries
print("\n3️⃣ Retrieval Test Ediliyor:\n")

test_queries = [
    "Kulübe nasıl üye olabilirim?",
    "Önümüzdeki hafta hangi etkinlikler var?",
    "Mentorluk programı hakkında bilgi ver",
    "Kulübün iletişim bilgileri neler?",
    "Girişimcilik kulübü ne zaman kuruldu?"
]

retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 5}
)

for i, query in enumerate(test_queries, 1):
    print(f"{i}. 🔍 Query: '{query}'")
    
    try:
        results = retriever.invoke(query)
        
        if results:
            print(f"   ✅ {len(results)} döküman bulundu:")
            for j, doc in enumerate(results, 1):
                source = doc.metadata.get('source', 'Unknown')
                score = doc.metadata.get('score', 'N/A')
                print(f"      {j}. {source}")
                print(f"         Preview: {doc.page_content[:100]}...")
                if score != 'N/A':
                    print(f"         Score: {score}")
        else:
            print("   ⚠️  Sonuç bulunamadı")
    
    except Exception as e:
        print(f"   ❌ Hata: {e}")
    
    print()

# 4. Similarity Search with Scores
print("\n4️⃣ Similarity Scores ile Arama:")
test_query = "Kulüp etkinlikleri"
print(f"   🔍 Query: '{test_query}'\n")

results_with_scores = vectorstore.similarity_search_with_score(
    test_query, 
    k=5
)

print(f"   📊 Top 5 Sonuç (Similarity Scores):\n")
for i, (doc, score) in enumerate(results_with_scores, 1):
    source = doc.metadata.get('source', 'Unknown')
    print(f"   {i}. Score: {score:.4f}")
    print(f"      Source: {source}")
    print(f"      Content: {doc.page_content[:150]}...")
    print()

print("=" * 60)
print("✅ RETRIEVAL TEST TAMAMLANDI!")
print("=" * 60)
print("\n💡 Değerlendirme:")
print("   - Doğru dökümanlar geldi mi?")
print("   - Score'lar mantıklı mı? (düşük = daha benzer)")
print("   - Türkçe karakterler düzgün görünüyor mu?")
print("\n📊 LangSmith'te trace'leri incele:")
print("   https://smith.langchain.com")