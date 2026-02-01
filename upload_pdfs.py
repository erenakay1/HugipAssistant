"""
PDF Yükleme Scripti
Belirtilen PDF'leri Pinecone'a yükler
"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv
load_dotenv()

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from src.core.config import get_settings
import os

def upload_pdfs(pdf_folder: str):
    """
    Klasördeki tüm PDF'leri Pinecone'a yükle
    
    Args:
        pdf_folder: PDF'lerin bulunduğu klasör yolu
    """
    settings = get_settings()
    
    print("\n" + "="*80)
    print("📤 PDF YÜKLEME İŞLEMİ")
    print("="*80 + "\n")
    
    # PDF klasörünü kontrol et
    pdf_path = Path(pdf_folder)
    if not pdf_path.exists():
        print(f"❌ Klasör bulunamadı: {pdf_folder}")
        return
    
    # PDF dosyalarını bul
    pdf_files = list(pdf_path.glob("*.pdf"))
    
    if not pdf_files:
        print(f"❌ Klasörde PDF dosyası bulunamadı: {pdf_folder}")
        return
    
    print(f"📄 Bulunan PDF'ler ({len(pdf_files)} adet):")
    for pdf in pdf_files:
        print(f"   - {pdf.name}")
    
    # Text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    
    # Embeddings
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        openai_api_key=settings.OPENAI_API_KEY
    )
    
    all_documents = []
    
    # Her PDF'i işle
    print("\n📖 PDF'ler işleniyor...")
    for pdf_file in pdf_files:
        print(f"\n   📄 {pdf_file.name}")
        
        try:
            # PDF'i yükle
            loader = PyPDFLoader(str(pdf_file))
            documents = loader.load()
            
            print(f"      ✓ {len(documents)} sayfa yüklendi")
            
            # Metadata ekle (kaynak ismi)
            for doc in documents:
                doc.metadata["source"] = pdf_file.stem  # .pdf uzantısı olmadan
            
            # Chunk'lara böl
            chunks = text_splitter.split_documents(documents)
            print(f"      ✓ {len(chunks)} chunk oluşturuldu")
            
            all_documents.extend(chunks)
            
        except Exception as e:
            print(f"      ❌ Hata: {str(e)}")
            continue
    
    if not all_documents:
        print("\n❌ İşlenecek döküman bulunamadı!")
        return
    
    print(f"\n📊 Toplam: {len(all_documents)} chunk yüklenecek")
    
    # Pinecone'a yükle
    print("\n🚀 Pinecone'a yükleniyor...")
    print("   (Bu işlem birkaç dakika sürebilir...)")
    
    try:
        vectorstore = PineconeVectorStore.from_documents(
            documents=all_documents,
            embedding=embeddings,
            index_name=settings.PINECONE_INDEX_NAME
        )
        
        print("\n✅ Yükleme tamamlandı!")
        
        # Test sorgusu
        print("\n🧪 Test sorgusu yapılıyor...")
        results = vectorstore.similarity_search("HUGİP nedir?", k=3)
        
        print(f"\n   Test sonucu: {len(results)} döküman bulundu")
        if results:
            print(f"   İlk sonuç: {results[0].metadata.get('source', 'Unknown')}")
        
    except Exception as e:
        print(f"\n❌ Yükleme hatası: {str(e)}")
        return
    
    print("\n" + "="*80)
    print("✅ İŞLEM BAŞARIYLA TAMAMLANDI")
    print("="*80)
    print("\n📋 Yüklenen Dökümanlar:")
    
    # Unique kaynak isimleri
    sources = set([doc.metadata.get('source', 'Unknown') for doc in all_documents])
    for source in sorted(sources):
        count = sum(1 for doc in all_documents if doc.metadata.get('source') == source)
        print(f"   - {source}: {count} chunk")
    
    print(f"\n📊 Toplam: {len(all_documents)} chunk yüklendi")

if __name__ == "__main__":
    # PDF klasörünü belirt
    pdf_folder = input("\nPDF klasör yolunu girin: ").strip()
    
    if not pdf_folder:
        # Default klasör
        pdf_folder = str(Path(__file__).parent / "data")
        print(f"\nDefault klasör kullanılıyor: {pdf_folder}")
    
    upload_pdfs(pdf_folder)