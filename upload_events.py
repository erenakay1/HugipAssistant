"""
Etkinlik PDF'ini Pinecone'a Yükleme
PDF dosyasını okur ve chunk'layarak yükler
"""
import os
import sys
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

load_dotenv()

print("=" * 60)
print("📚 Etkinlik PDF'i Pinecone'a Yükleniyor")
print("=" * 60)

# Config
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "hugip-doc-index")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# PDF dosyası
pdf_file = "./documents/etkinlikler.pdf"

# Dosya var mı kontrol et
if not os.path.exists(pdf_file):
    print(f"\n❌ '{pdf_file}' dosyası bulunamadı!")
    print("\n💡 Lütfen etkinlikler.pdf dosyasını proje klasörüne kopyala")
    print("   ve bu scripti tekrar çalıştır.")
    sys.exit(1)

# 1. PDF'i yükle
print(f"\n1️⃣ PDF yükleniyor: {pdf_file}")
try:
    loader = PyPDFLoader(pdf_file)
    documents = loader.load()
    print(f"   ✅ {len(documents)} sayfa yüklendi")
    
    # Toplam karakter sayısı
    total_chars = sum(len(doc.page_content) for doc in documents)
    print(f"   📄 Toplam karakter: {total_chars}")
except Exception as e:
    print(f"   ❌ PDF okuma hatası: {e}")
    print("\n💡 PyPDF kütüphanesi yüklü değil olabilir:")
    print("   pip install pypdf")
    sys.exit(1)

# 2. Metadata ekle
print("\n2️⃣ Metadata ekleniyor...")
for i, doc in enumerate(documents):
    doc.metadata = {
        "source": pdf_file,
        "page": i + 1,
        "category": "etkinlikler",
        "type": "event_information",
        "language": "tr",
        "updated": "2024-12-04"
    }
print(f"   ✅ {len(documents)} sayfa için metadata eklendi")

# 3. Chunking
print("\n3️⃣ Chunking yapılıyor...")
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,  # Etkinlik bilgileri için küçük chunk
    chunk_overlap=150,
    separators=["\n\n", "\n", ". ", " ", ""],
    length_function=len,
)

chunks = text_splitter.split_documents(documents)
print(f"   ✅ {len(chunks)} chunk oluşturuldu")

# Örnek chunk göster
if chunks:
    print("\n   📄 Örnek chunk (ilk 200 karakter):")
    example_text = chunks[0].page_content[:200].replace('\n', ' ')
    print(f"   {example_text}...")

# 4. Embeddings
print("\n4️⃣ Embeddings hazırlanıyor...")
try:
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        openai_api_key=OPENAI_API_KEY
    )
    print("   ✅ Embeddings hazır")
except Exception as e:
    print(f"   ❌ Embeddings hatası: {e}")
    print("\n💡 OpenAI API key'i kontrol et (.env dosyası)")
    sys.exit(1)

# 5. Pinecone'a yükle
print("\n5️⃣ Pinecone'a yükleniyor...")
print("   ⏳ Bu işlem 1-2 dakika sürebilir...")

try:
    vectorstore = PineconeVectorStore.from_documents(
        documents=chunks,
        embedding=embeddings,
        index_name=PINECONE_INDEX_NAME
    )
    print(f"   ✅ {len(chunks)} chunk başarıyla yüklendi!")
except Exception as e:
    print(f"   ❌ Pinecone yükleme hatası: {e}")
    print("\n💡 Olası sebepler:")
    print("   - Pinecone API key yanlış (.env dosyasını kontrol et)")
    print("   - Internet bağlantısı yok")
    print("   - Index adı yanlış (hugip-doc-index olmalı)")
    sys.exit(1)

# 6. Test retrieval
print("\n6️⃣ Retrieval test ediliyor...")

test_queries = [
    "Social Media Talks etkinliği hakkında bilgi ver",
    "DigitalMAG etkinliğine kaç kişi katıldı?",
    "FESTUP ne zaman yapılıyor?",
    "HUGİP Akademi'de ne öğretiliyor?",
    "Etkinliklere nasıl katılabilirim?",
]

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

for query in test_queries:
    print(f"\n   🔍 '{query}'")
    try:
        results = retriever.invoke(query)
        if results:
            print(f"   ✅ {len(results)} sonuç bulundu")
            # İlk sonucu göster
            first_result = results[0].page_content[:120].replace('\n', ' ')
            print(f"   📄 İlk sonuç: {first_result}...")
        else:
            print("   ⚠️  Sonuç bulunamadı")
    except Exception as e:
        print(f"   ❌ Retrieval hatası: {e}")

print("\n" + "=" * 60)
print("✅ ETKİNLİK PDF'İ BAŞARIYLA YÜKLENDİ!")
print("=" * 60)
print("\n📊 Özet:")
print(f"   - Kaynak dosya: {pdf_file}")
print(f"   - Sayfa sayısı: {len(documents)}")
print(f"   - Toplam chunk: {len(chunks)}")
print(f"   - Index: {PINECONE_INDEX_NAME}")
print(f"   - Embedding model: text-embedding-3-small")
print("\n💡 Artık etkinlikler hakkında sorular sorabilirsiniz!")
print("\n🔥 Sonraki adım:")
print("   python test_modular.py")
print("   (veya test_events.py ile detaylı test)")