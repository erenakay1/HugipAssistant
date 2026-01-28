"""
Reflection System Test
Hallucination detection ve regeneration'ı test eder
"""
import sys
import os
from pathlib import Path

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv
load_dotenv()

from src.graph import build_graph, GraphState

print("=" * 70)
print("🔍 Reflection System Test - Hallucination Detection")
print("=" * 70)

# Graph oluştur
print("\n1️⃣ Graph oluşturuluyor (Reflection ile)...")
app = build_graph()
print("   ✅ Graph başarıyla oluşturuldu!")

# Test soruları (Hallucination riski yüksek)
test_questions = [
    # Tarih içeren sorular (Hallucination riski yüksek)
    "Social Media Talks ne zaman yapılacak?",
    "FESTUP hangi tarihte?",
    
    # İsim içeren sorular (Hallucination riski yüksek)
    "Social Media Talks'ta kimler konuşacak?",
    "Melih Abuaf kimdir?",
    
    # Sayı içeren sorular
    "FESTUP'a kaç kişi katılıyor?",
    "DigitalMAG'da kaç marka var?",
    
    # Genel sorular (Düşük risk)
    "Kulübün amacı nedir?",
    "FESTUP nedir?",
]

print("\n" + "=" * 70)
print("2️⃣ Reflection Test Başlıyor:")
print("=" * 70)

for i, question in enumerate(test_questions, 1):
    print(f"\n{'='*70}")
    print(f"Test {i}/{len(test_questions)}: '{question}'")
    print("=" * 70)
    
    # Initial state
    initial_state: GraphState = {
        "question": question,
        "generation": "",
        "documents": [],
        "decision": "",
        "web_results": [],
        "iterations": 0
    }
    
    # Run graph
    result = app.invoke(initial_state)
    
    # Route
    route_emoji = {
        "rag": "📚",
        "web_search": "🌐",
        "direct": "💬"
    }
    print(f"\n🔀 Route: {route_emoji.get(result['decision'], '❓')} {result['decision'].upper()}")
    
    # Iterations (Reflection loop sayısı)
    print(f"🔄 Reflection Iterations: {result.get('iterations', 0)}")
    if result.get('iterations', 0) > 1:
        print(f"   ⚠️  Regeneration yapıldı! (Quality issue detected)")
    
    # Cevap
    print(f"\n📝 FINAL CEVAP:")
    print("-" * 70)
    print(result['generation'])
    print("-" * 70)
    
    # Kaynaklar
    if result['documents']:
        print(f"\n📚 Kaynaklar ({len(result['documents'])} döküman):")
        sources = set()
        for doc in result['documents']:
            source = doc.metadata.get('source', 'Unknown')
            source = source.split('\\')[-1] if '\\' in source else source.split('/')[-1]
            sources.add(source)
        for j, source in enumerate(sources, 1):
            print(f"   {j}. {source}")
    
    # Kısa bekleme
    if i < len(test_questions):
        import time
        time.sleep(1)

print("\n" + "=" * 70)
print("✅ REFLECTION TEST TAMAMLANDI!")
print("=" * 70)
print("\n📊 Değerlendirme Kriterleri:")
print("   ✅ Tarih bilgileri doğru mu?")
print("   ✅ İsimler doğru mu?")
print("   ✅ Sayılar doğru mu?")
print("   ✅ Regeneration gerekti mi?")
print("   ✅ Final cevap kaliteli mi?")
print("\n🔗 LangSmith'te detaylı incele:")
print("   - Hallucination grader sonuçlarını gör")
print("   - Regeneration trace'lerini incele")
print("   - Reasoning'leri oku")
print("   https://smith.langchain.com")