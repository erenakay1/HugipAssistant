"""
Conversation Memory Test
Multi-turn conversation'ları test eder
"""
import sys
import os
from pathlib import Path
import uuid

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv
load_dotenv()

from src.graph import build_graph, GraphState
from src.services.memory_service import MemoryService

print("=" * 70)
print("🧠 Conversation Memory Test - Multi-turn Conversations")
print("=" * 70)

# Graph ve Memory service
print("\n1️⃣ Graph ve Memory oluşturuluyor...")
app = build_graph()
memory_service = MemoryService()
print("   ✅ Hazır!")

# Test scenarios (Multi-turn conversations)
test_conversations = [
    {
        "name": "FESTUP Conversation",
        "turns": [
            "FESTUP nedir?",
            "Ne zaman yapılıyor?",  # "Ne" = FESTUP (memory'den)
            "Kaç kişi katılıyor?",  # "Kaç kişi" = FESTUP'a
            "İş bulabilir miyim?",  # FESTUP'ta
        ]
    },
    {
        "name": "Social Media Talks Conversation",
        "turns": [
            "Social Media Talks hakkında bilgi ver",
            "Kimler konuşacak?",  # Social Media Talks'ta
            "Ne zaman?",  # Social Media Talks
            "Nerede yapılacak?",  # Social Media Talks
        ]
    },
    {
        "name": "Kulüp ve Üyelik",
        "turns": [
            "Kulübün amacı nedir?",
            "Nasıl üye olabilirim?",  # Kulübe
            "Üyelik ücreti var mı?",  # Kulüp üyeliği
        ]
    }
]

print("\n" + "=" * 70)
print("2️⃣ Multi-turn Conversation Testleri:")
print("=" * 70)

for conv_idx, conversation in enumerate(test_conversations, 1):
    # Her conversation için yeni session
    session_id = str(uuid.uuid4())
    
    print(f"\n{'='*70}")
    print(f"💬 Conversation {conv_idx}: {conversation['name']}")
    print(f"Session ID: {session_id}")
    print("=" * 70)
    
    for turn_idx, question in enumerate(conversation["turns"], 1):
        print(f"\n🔹 Turn {turn_idx}: '{question}'")
        print("-" * 70)
        
        
        
        # Initial state
        initial_state: GraphState = {
            "question": question,
            "generation": "",
            "documents": [],
            "decision": "",
            "web_results": [],
            "iterations": 0,
            "session_id": session_id
        }
        
        # Run graph
        result = app.invoke(initial_state)
        
        # Route
        route_emoji = {
            "rag": "📚",
            "web_search": "🌐",
            "direct": "💬"
        }
        print(f"🔀 Route: {route_emoji.get(result['decision'], '❓')} {result['decision'].upper()}")
        
        # Cevap
        print(f"\n📝 CEVAP:")
        print(result['generation'][:200] + "..." if len(result['generation']) > 200 else result['generation'])
        
        
        
        # Memory context göster (debug için)
        if turn_idx > 1:
            context = memory_service.get_context(session_id, last_n=4)
            print(f"\n💭 Memory Context:")
            for line in context.split('\n')[-2:]:  # Son 2 satır
                print(f"   {line[:80]}..." if len(line) > 80 else f"   {line}")
        
        print("-" * 70)
    
    # Conversation summary
    print(f"\n📊 Conversation Summary:")
    print(f"   Toplam turn: {len(conversation['turns'])}")
    print(f"   Son topic: {memory_service.get_last_topic(session_id)}")
    
    # Session temizle
    memory_service.clear(session_id)

print("\n" + "=" * 70)
print("✅ MEMORY TEST TAMAMLANDI!")
print("=" * 70)
print("\n📊 Değerlendirme:")
print("   ✅ 'Ne zaman?' → Önceki topic'i hatırladı mı?")
print("   ✅ 'Kimler?' → Etkinlik context'ini anladı mı?")
print("   ✅ 'Kaç kişi?' → FESTUP'u hatırladı mı?")
print("   ✅ Router doğru route etti mi?")
print("\n🔗 LangSmith'te incele:")
print("   - Router'a giden history'yi gör")
print("   - Context-aware routing'i analiz et")
print("   https://smith.langchain.com")