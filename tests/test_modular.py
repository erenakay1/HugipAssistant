"""
Modüler Yapı Test
Gerçek proje yapısıyla test
"""
import sys
import os
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv
load_dotenv()

from src.graph import build_graph, GraphState

print("=" * 60)
print("🤖 Kulüp Asistanı - Modüler Yapı Test")
print("=" * 60)

# Graph oluştur
print("\n1️⃣ Graph oluşturuluyor...")
app = build_graph()
print("   ✅ Graph başarıyla oluşturuldu!")

# Test soruları
test_questions = [
    # Genel
    "Merhaba!",
    "Kulübün amacı nedir?",
    "Yönetim kurulu nasıl seçilir?",
    "Dış İlişkiler ekibi ne yapar?",
    
    # Etkinlikler (YENİ!)
    "Social Media Talks etkinliği hakkında bilgi ver",
    "Social Media Talks'ta kimler konuşacak?",
    "Melih Aktaş kimdir?",
    "DigitalMAG ne zaman yapılıyor?",
    "FESTUP'ta iş bulabilir miyim?",
    "HUGİP Akademi'de hangi eğitimler var?",
    
    # Web Search
    "Yapay zeka hakkında bilgi ver",
]

print("\n" + "=" * 60)
print("2️⃣ Test Başlıyor (11 Soru):")
print("=" * 60)

for i, question in enumerate(test_questions, 1):
    print(f"\n{'='*60}")
    print(f"Test {i}: '{question}'")
    print("=" * 60)
    
    # Initial state
    initial_state: GraphState = {
        "question": question,
        "generation": "",
        "documents": [],
        "decision": "",
        "web_results": [],
        "iterations": 0  # YENİ!
    }
    
    # Run graph
    result = app.invoke(initial_state)
    
    # Route göster
    route_emoji = {
        "rag": "📚",
        "web_search": "🌐",
        "direct": "💬"
    }
    print(f"\n🔀 Route: {route_emoji.get(result['decision'], '❓')} {result['decision'].upper()}")
    
    # Cevap
    print(f"\n📝 CEVAP:")
    print("-" * 60)
    print(result['generation'])
    print("-" * 60)
    
    # Kaynaklar (varsa)
    if result['documents']:
        print(f"\n📚 Kaynak Dökümanlar ({len(result['documents'])}):")
        sources = set()
        for doc in result['documents']:
            source = doc.metadata.get('source', 'Unknown')
            # Sadece dosya adını al
            source = source.split('\\')[-1] if '\\' in source else source.split('/')[-1]
            sources.add(source)
        for j, source in enumerate(sources, 1):
            print(f"   {j}. {source}")

print("\n" + "=" * 60)
print("✅ TEST TAMAMLANDI!")
print("=" * 60)
print("\n🎉 Modüler yapı çalışıyor!")
print("\n📂 Proje Yapısı:")
print("""
club-assistant-service/
├── src/
│   ├── core/
│   │   ├── __init__.py
│   │   └── config.py ✅
│   ├── services/
│   │   ├── __init__.py
│   │   ├── llm_service.py ✅
│   │   └── vectorstore_service.py ✅
│   ├── graph/
│   │   ├── __init__.py
│   │   ├── state.py ✅
│   │   ├── graph.py ✅
│   │   └── nodes/
│   │       ├── __init__.py
│   │       ├── router.py ✅
│   │       ├── rag.py ✅
│   │       └── generation.py ✅
│   └── __init__.py
├── test_modular.py ✅
└── .env
""")
print("\n🔗 LangSmith:")
print(f"   https://smith.langchain.com")