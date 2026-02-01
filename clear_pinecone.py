"""
Pinecone Temizleme Scripti
Tüm dökümanları siler
"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent  # tests klasöründen çıkmak için .parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from dotenv import load_dotenv
load_dotenv()

from pinecone import Pinecone
from src.core.config import get_settings

def clear_pinecone():
    """Pinecone index'ini tamamen temizle"""
    settings = get_settings()
    
    # Pinecone client
    pc = Pinecone(api_key=settings.PINECONE_API_KEY)
    index = pc.Index(settings.PINECONE_INDEX_NAME)
    
    print("\n" + "="*80)
    print("⚠️  PINECONE TEMİZLEME - DİKKAT!")
    print("="*80 + "\n")
    
    # Index stats
    stats = index.describe_index_stats()
    total_vectors = stats.total_vector_count
    
    print(f"📊 Mevcut vektör sayısı: {total_vectors}")
    
    if total_vectors == 0:
        print("\n✅ Index zaten boş!")
        return
    
    # Onay al
    print("\n⚠️  TÜM DÖKÜMANLAR SİLİNECEK!")
    confirm = input("\nDevam etmek için 'EVET' yazın: ")
    
    if confirm != "EVET":
        print("\n❌ İşlem iptal edildi.")
        return
    
    print("\n🗑️  Tüm vektörler siliniyor...")
    
    try:
        # Default namespace'i sil (namespace parametresi olmadan)
        print(f"   Default namespace siliniyor...")
        index.delete(delete_all=True)
        print(f"   ✓ Default namespace silindi")
    except Exception as e:
        if "not found" in str(e).lower() or "404" in str(e):
            print(f"   ⓘ Default namespace zaten boş")
        else:
            print(f"   ⚠️  Hata: {str(e)}")
    
    # Diğer namespace'leri sil
    if stats.namespaces:
        for namespace in stats.namespaces.keys():
            if namespace:  # Boş string değilse
                try:
                    print(f"   Namespace siliniyor: '{namespace}'")
                    index.delete(delete_all=True, namespace=namespace)
                    print(f"   ✓ '{namespace}' silindi")
                except Exception as e:
                    if "not found" in str(e).lower() or "404" in str(e):
                        print(f"   ⓘ '{namespace}' zaten boş")
                    else:
                        print(f"   ⚠️  Hata: {str(e)}")
    
    print("\n✅ Pinecone index temizlendi!")
    
    # Final stats
    import time
    time.sleep(2)  # Pinecone'un güncellenmesini bekle
    
    final_stats = index.describe_index_stats()
    print(f"\n📊 Kalan vektör sayısı: {final_stats.total_vector_count}")
    
    print("\n" + "="*80)
    print("✅ İŞLEM TAMAMLANDI")
    print("="*80)

if __name__ == "__main__":
    clear_pinecone()