"""
Memory Service
Conversation history'yi yönetir (session-based)
"""
from typing import List, Dict, Optional
from datetime import datetime

class ConversationMemory:
    """
    Basit in-memory conversation storage
    
    Production'da Redis veya DB kullanılmalı.
    Şimdilik dictionary ile in-memory.
    """
    
    def __init__(self, max_history: int = 10):
        """
        Args:
            max_history: Kaç mesaj saklanacak (default: 10 = 5 user + 5 assistant)
        """
        # session_id -> messages
        self.sessions: Dict[str, List[Dict]] = {}
        self.max_history = max_history
    
    def add_message(
        self, 
        session_id: str, 
        role: str, 
        content: str,
        metadata: Optional[Dict] = None
    ):
        """
        Yeni mesaj ekle (duplicate prevention ile)
        
        Args:
            session_id: Session identifier
            role: 'user' veya 'assistant'
            content: Mesaj içeriği
            metadata: Ek bilgiler (route, sources, vb.)
        """
        if session_id not in self.sessions:
            self.sessions[session_id] = []
        
        # Duplicate prevention: Son mesaj aynı mı kontrol et
        if self.sessions[session_id]:
            last_message = self.sessions[session_id][-1]
            if (last_message["role"] == role and 
                last_message["content"] == content):
                # Aynı mesaj, ekleme!
                return
        
        message = {
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata or {}
        }
        
        self.sessions[session_id].append(message)
        
        # Max history'yi aş, eski mesajları sil
        if len(self.sessions[session_id]) > self.max_history:
            self.sessions[session_id] = self.sessions[session_id][-self.max_history:]
    
    def get_history(
        self, 
        session_id: str, 
        last_n: Optional[int] = None
    ) -> List[Dict]:
        """
        Session'ın message history'sini getir
        
        Args:
            session_id: Session identifier
            last_n: Son N mesaj (None = hepsi)
            
        Returns:
            List of messages
        """
        if session_id not in self.sessions:
            return []
        
        messages = self.sessions[session_id]
        
        if last_n:
            return messages[-last_n:]
        
        return messages
    
    def get_context_string(
        self, 
        session_id: str, 
        last_n: int = 6
    ) -> str:
        """
        Context string olarak history
        LLM'e prompt'ta vereceğiz
        
        Args:
            session_id: Session identifier
            last_n: Son N mesaj (default: 6 = 3 user + 3 assistant)
            
        Returns:
            Formatted conversation history
        """
        messages = self.get_history(session_id, last_n)
        
        if not messages:
            return ""
        
        context_lines = []
        for msg in messages:
            role_label = "Kullanıcı" if msg["role"] == "user" else "Asistan"
            # Mesajı kısa tut (max 100 karakter)
            content = msg['content'][:150] + "..." if len(msg['content']) > 150 else msg['content']
            context_lines.append(f"{role_label}: {content}")
        
        return "\n".join(context_lines)
    
    def clear_session(self, session_id: str):
        """Session'ı temizle"""
        if session_id in self.sessions:
            del self.sessions[session_id]
    
    def get_last_topic(self, session_id: str) -> Optional[str]:
        """
        Son konuşulan topic'i çıkar (basit heuristic)
        
        Returns:
            Son bahsedilen önemli kelime/topic
        """
        messages = self.get_history(session_id, last_n=6)
        
        if not messages:
            return None
        
        # Tüm mesajlara bak (user + assistant)
        all_content = " ".join([m["content"] for m in messages])
        
        # Basit keyword extraction
        # Öncelik sırasına göre kontrol et
        keywords = [
            ("FESTUP", "FESTUP"),
            ("Social Media Talks", "Social Media Talks"),
            ("DigitalMAG", "DigitalMAG"),
            ("HUGİP Akademi", "HUGİP Akademi"),
            ("üyelik", "üyelik"),
            ("üye ol", "üyelik"),
            ("yönetim kurulu", "yönetim"),
            ("kulüp", "kulüp"),
            ("etkinlik", "etkinlik"),
        ]
        
        for keyword, topic in keywords:
            if keyword.lower() in all_content.lower():
                return topic
        
        return None


class MemoryService:
    """
    Memory Service Wrapper
    Singleton pattern ile tek instance
    """
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(MemoryService, cls).__new__(cls)
            cls._instance.memory = ConversationMemory(max_history=10)
        return cls._instance
    
    def add_user_message(self, session_id: str, content: str):
        """User mesajı ekle"""
        self.memory.add_message(session_id, "user", content)
    
    def add_assistant_message(
        self, 
        session_id: str, 
        content: str,
        route: Optional[str] = None,
        sources: Optional[List[str]] = None
    ):
        """Assistant mesajı ekle"""
        metadata = {}
        if route:
            metadata["route"] = route
        if sources:
            metadata["sources"] = sources
        
        self.memory.add_message(session_id, "assistant", content, metadata)
    
    def get_context(self, session_id: str, last_n: int = 6) -> str:
        """Context string"""
        return self.memory.get_context_string(session_id, last_n)
    
    def get_last_topic(self, session_id: str) -> Optional[str]:
        """Son topic"""
        return self.memory.get_last_topic(session_id)
    
    def clear(self, session_id: str):
        """Session temizle"""
        self.memory.clear_session(session_id)