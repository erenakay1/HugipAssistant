"""
Pinecone Setup & Document Ingestion
PDF'leri yüklemek için kullan
"""
import os
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_community.document_loaders import (
    PyPDFLoader,
    DirectoryLoader,
    TextLoader
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pathlib import Path

load_dotenv()

print("=" * 60)
print("📦 PINECONE SETUP & DOCUMENT INGESTION")
print("=" * 60)

# Config
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "hugip-doc-index")  # Senin index'in
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not PINECONE_API_KEY or not OPENAI_API_KEY:
    print("❌ API keys eksik! .env dosyasını kontrol et")
    exit(1)

# 1. Pinecone'a Bağlan
print("\n1️⃣ Pinecone'a bağlanılıyor...")

pc = Pinecone(api_key=PINECONE_API_KEY)

# Index'i kontrol et
try:
    index_info = pc.describe_index(PINECONE_INDEX_NAME)
    print(f"   ✅ Index bulundu: {PINECONE_INDEX_NAME}")
    print(f"   📊 Dimension: {index_info.dimension}")
    print(f"   🌐 Region: {index_info.spec.serverless.region}")
except Exception as e:
    print(f"   ❌ Index bulunamadı: {e}")
    print(f"   💡 Lütfen Pinecone dashboard'undan index oluştur")
    exit(1)

# 2. Embeddings
print("\n2️⃣ OpenAI Embeddings hazırlanıyor...")
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",  # Senin index'indeki model
    openai_api_key=OPENAI_API_KEY
)
print("   ✅ Embeddings hazır")

# 3. Document Loading
print("\n3️⃣ Dökümanlar yükleniyor...")

def load_documents_from_directory(directory: str):
    """Bir klasördeki tüm PDF ve TXT dosyalarını yükle"""
    documents = []
    path = Path(directory)
    
    if not path.exists():
        print(f"   ⚠️  Klasör bulunamadı: {directory}")
        return documents
    
    # PDF loader
    try:
        pdf_loader = DirectoryLoader(
            directory,
            glob="**/*.pdf",
            loader_cls=PyPDFLoader,
            show_progress=True
        )
        pdf_docs = pdf_loader.load()
        documents.extend(pdf_docs)
        print(f"   📄 {len(pdf_docs)} PDF yüklendi")
    except Exception as e:
        print(f"   ⚠️  PDF yükleme hatası: {e}")
    
    # TXT loader
    try:
        txt_loader = DirectoryLoader(
            directory,
            glob="**/*.txt",
            loader_cls=TextLoader,
            show_progress=True
        )
        txt_docs = txt_loader.load()
        documents.extend(txt_docs)
        print(f"   📝 {len(txt_docs)} TXT yüklendi")
    except Exception as e:
        print(f"   ⚠️  TXT yükleme hatası: {e}")
    
    return documents

# Dökümanları yükle (örnek klasör: ./documents)
documents_dir = "./documents"
documents = load_documents_from_directory(documents_dir)

if not documents:
    print(f"\n   ⚠️  '{documents_dir}' klasöründe döküman bulunamadı!")
    print(f"   💡 Lütfen PDF/TXT dosyalarını bu klasöre ekle ve tekrar çalıştır")
    print(f"\n   Örnek klasör yapısı:")
    print(f"   ./documents/")
    print(f"   ├── kulup_hakkinda.pdf")
    print(f"   ├── uyelik_bilgileri.pdf")
    print(f"   └── etkinlik_takvimi.txt")
    exit(0)

print(f"\n   ✅ Toplam {len(documents)} döküman yüklendi")

# 4. Text Splitting
print("\n4️⃣ Dökümanlar chunking yapılıyor...")

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,  # Her chunk 1000 karakter
    chunk_overlap=200,  # 200 karakter overlap
    length_function=len,
    add_start_index=True,
)

chunks = text_splitter.split_documents(documents)
print(f"   ✅ {len(chunks)} chunk oluşturuldu")

# 5. Pinecone'a Yükle
print("\n5️⃣ Pinecone'a yükleniyor...")
print("   ⏳ Bu işlem biraz sürebilir...")

try:
    vectorstore = PineconeVectorStore.from_documents(
        documents=chunks,
        embedding=embeddings,
        index_name=PINECONE_INDEX_NAME
    )
    print(f"   ✅ {len(chunks)} chunk Pinecone'a yüklendi!")
except Exception as e:
    print(f"   ❌ Yükleme hatası: {e}")
    exit(1)

# 6. Test Retrieval
print("\n6️⃣ Retrieval test ediliyor...")

test_query = "Kulübe nasıl üye olabilirim?"
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
results = retriever.invoke(test_query)

print(f"\n   🔍 Test Query: '{test_query}'")
print(f"   📚 {len(results)} döküman bulundu:")
for i, doc in enumerate(results, 1):
    source = doc.metadata.get('source', 'Unknown')
    print(f"      {i}. {source}")
    print(f"         {doc.page_content[:100]}...")

print("\n" + "=" * 60)
print("✅ PINECONE SETUP TAMAMLANDI!")
print("=" * 60)
print(f"\n📊 Özet:")
print(f"   - Index: {PINECONE_INDEX_NAME}")
print(f"   - Döküman sayısı: {len(documents)}")
print(f"   - Chunk sayısı: {len(chunks)}")
print(f"   - Embedding model: text-embedding-ada-002")
print(f"\n🔗 Pinecone Dashboard:")
print(f"   https://app.pinecone.io")