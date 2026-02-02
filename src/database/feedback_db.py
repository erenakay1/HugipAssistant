"""
Database Setup for Feedback System
Uses Supabase Python client (no psycopg2 needed!)
Falls back to SQLite for local development
"""
import os
import sqlite3
from datetime import datetime, timedelta
from typing import Optional, List, Dict
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Supabase client
try:
    from supabase import create_client, Client
    SUPABASE_AVAILABLE = True
except ImportError:
    SUPABASE_AVAILABLE = False

class FeedbackDB:
    """Feedback database manager - Supabase or SQLite"""
    
    def __init__(self, db_path: str = "feedback.db"):
        self.supabase_url = os.getenv("SUPABASE_URL")
        self.supabase_key = os.getenv("SUPABASE_KEY")
        
        if self.supabase_url and self.supabase_key and SUPABASE_AVAILABLE:
            self.db_type = "supabase"
            self.client: Client = create_client(self.supabase_url, self.supabase_key)
            print("📊 Using Supabase database")
        else:
            self.db_type = "sqlite"
            self.db_path = db_path
            print(f"📊 Using SQLite database: {db_path}")
            self.init_sqlite()
    
    # ==================== SQLite ====================
    def init_sqlite(self):
        """Create SQLite tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                question TEXT NOT NULL,
                answer TEXT NOT NULL,
                route TEXT,
                sources TEXT,
                rating TEXT NOT NULL,
                issue_type TEXT,
                comment TEXT,
                user_email TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS chat_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                question TEXT NOT NULL,
                answer TEXT NOT NULL,
                route TEXT,
                sources TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        conn.commit()
        conn.close()
    
    # ==================== ADD FEEDBACK ====================
    def add_feedback(
        self,
        session_id: str,
        question: str,
        answer: str,
        rating: str,
        route: Optional[str] = None,
        sources: Optional[str] = None,
        issue_type: Optional[str] = None,
        comment: Optional[str] = None,
        user_email: Optional[str] = None
    ) -> int:
        """Add feedback"""
        if self.db_type == "supabase":
            data = {
                "session_id": session_id,
                "question": question,
                "answer": answer,
                "rating": rating,
                "route": route or "",
                "sources": sources or "",
                "issue_type": issue_type or "",
                "comment": comment or "",
                "user_email": user_email or ""
            }
            response = self.client.table("feedback").insert(data).execute()
            return response.data[0]["id"] if response.data else 0
        else:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO feedback (session_id, question, answer, rating, route, sources, issue_type, comment, user_email)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (session_id, question, answer, rating, route, sources, issue_type, comment, user_email))
            feedback_id = cursor.lastrowid
            conn.commit()
            conn.close()
            return feedback_id
    
    # ==================== GET ALL FEEDBACK ====================
    def get_all_feedback(self, limit: int = 100) -> List[Dict]:
        """Get all feedback"""
        if self.db_type == "supabase":
            response = (
                self.client.table("feedback")
                .select("*")
                .order("created_at", desc=True)
                .limit(limit)
                .execute()
            )
            return response.data
        else:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM feedback ORDER BY created_at DESC LIMIT ?", (limit,))
            rows = [dict(row) for row in cursor.fetchall()]
            conn.close()
            return rows
    
    # ==================== GET FEEDBACK STATS ====================
    def get_feedback_stats(self) -> Dict:
        """Get feedback statistics"""
        feedback_list = self.get_all_feedback(limit=10000)
        
        total = len(feedback_list)
        ratings = {}
        
        for item in feedback_list:
            rating = item.get("rating", "")
            ratings[rating] = ratings.get(rating, 0) + 1
        
        positive = ratings.get("positive", 0)
        avg_rating = round((positive / total) * 100, 1) if total > 0 else 0
        
        return {
            "total": total,
            "ratings": ratings,
            "avg_rating": avg_rating
        }
    
    # ==================== CHAT HISTORY ====================
    def add_chat_history(
        self,
        session_id: str,
        question: str,
        answer: str,
        route: Optional[str] = None,
        sources: Optional[str] = None,
        response_time: Optional[float] = None
    ) -> int:
        """Add chat to history"""
        if self.db_type == "supabase":
            data = {
                "session_id": session_id,
                "question": question,
                "answer": answer,
                "route": route or "",
                "sources": sources or "",
                "response_time": response_time or 0.0
            }
            response = self.client.table("chat_history").insert(data).execute()
            return response.data[0]["id"] if response.data else 0
        else:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO chat_history (session_id, question, answer, route, sources, response_time)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (session_id, question, answer, route, sources, response_time or 0.0))
            chat_id = cursor.lastrowid
            conn.commit()
            conn.close()
            return chat_id
    
    def get_chat_history(self, session_id: str, limit: int = 50) -> List[Dict]:
        """Get chat history for a session"""
        if self.db_type == "supabase":
            response = (
                self.client.table("chat_history")
                .select("*")
                .eq("session_id", session_id)
                .order("created_at", desc=True)
                .limit(limit)
                .execute()
            )
            return response.data
        else:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM chat_history WHERE session_id = ? ORDER BY created_at DESC LIMIT ?",
                (session_id, limit)
            )
            rows = [dict(row) for row in cursor.fetchall()]
            conn.close()
            return rows
    
    def get_total_chats(self) -> int:
        """Get total chats"""
        if self.db_type == "supabase":
            response = self.client.table("chat_history").select("count", count="exact").execute()
            return response.count or 0
        else:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM chat_history")
            total = cursor.fetchone()[0]
            conn.close()
            return total
    
    def get_recent_feedback(self, days: int = 7) -> List[Dict]:
        """Get feedback from last N days"""
        if self.db_type == "supabase":
            cutoff = (datetime.now() - timedelta(days=days)).isoformat()
            response = (
                self.client.table("feedback")
                .select("*")
                .gte("created_at", cutoff)
                .order("created_at", desc=True)
                .execute()
            )
            return response.data
        else:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM feedback
                WHERE created_at >= datetime('now', '-' || ? || ' days')
                ORDER BY created_at DESC
            """, (days,))
            rows = [dict(row) for row in cursor.fetchall()]
            conn.close()
            return rows