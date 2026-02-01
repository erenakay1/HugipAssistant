"""
Database Setup for Feedback System
SQLite database for storing user feedback
"""
import sqlite3
from datetime import datetime
from typing import Optional, List, Dict
from pathlib import Path

class FeedbackDB:
    """Feedback database manager"""
    
    def __init__(self, db_path: str = "feedback.db"):
        """
        Initialize database connection
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        self.init_db()
    
    def init_db(self):
        """Create tables if they don't exist"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Feedback table
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
        
        # Chat history table (opsiyonel - analytics için)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS chat_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                question TEXT NOT NULL,
                answer TEXT NOT NULL,
                route TEXT,
                sources TEXT,
                response_time REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        conn.commit()
        conn.close()
    
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
        """
        Add feedback to database
        
        Args:
            session_id: Session identifier
            question: User question
            answer: Assistant answer
            rating: 'positive' or 'negative'
            route: Router decision (rag/direct/web_search)
            sources: Source documents (comma-separated)
            issue_type: Type of issue (if negative)
            comment: User comment
            user_email: User email (optional)
            
        Returns:
            Feedback ID
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO feedback 
            (session_id, question, answer, route, sources, rating, issue_type, comment, user_email)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (session_id, question, answer, route, sources, rating, issue_type, comment, user_email))
        
        feedback_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        return feedback_id
    
    def log_chat(
        self,
        session_id: str,
        question: str,
        answer: str,
        route: str,
        sources: Optional[str] = None,
        response_time: Optional[float] = None
    ):
        """Log chat interaction (for analytics)"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO chat_history 
            (session_id, question, answer, route, sources, response_time)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (session_id, question, answer, route, sources, response_time))
        
        conn.commit()
        conn.close()
    
    def get_all_feedback(self, limit: int = 100) -> List[Dict]:
        """Get all feedback (for admin panel)"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT * FROM feedback 
            ORDER BY created_at DESC 
            LIMIT ?
        """, (limit,))
        
        rows = cursor.fetchall()
        conn.close()
        
        return [dict(row) for row in rows]
    
    def get_negative_feedback(self, limit: int = 50) -> List[Dict]:
        """Get negative feedback only"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT * FROM feedback 
            WHERE rating = 'negative'
            ORDER BY created_at DESC 
            LIMIT ?
        """, (limit,))
        
        rows = cursor.fetchall()
        conn.close()
        
        return [dict(row) for row in rows]
    
    def get_feedback_stats(self) -> Dict:
        """Get feedback statistics"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Total feedback
        cursor.execute("SELECT COUNT(*) FROM feedback")
        total = cursor.fetchone()[0]
        
        # Positive feedback
        cursor.execute("SELECT COUNT(*) FROM feedback WHERE rating = 'positive'")
        positive = cursor.fetchone()[0]
        
        # Negative feedback
        cursor.execute("SELECT COUNT(*) FROM feedback WHERE rating = 'negative'")
        negative = cursor.fetchone()[0]
        
        # Issue types breakdown
        cursor.execute("""
            SELECT issue_type, COUNT(*) as count 
            FROM feedback 
            WHERE rating = 'negative' AND issue_type IS NOT NULL
            GROUP BY issue_type
            ORDER BY count DESC
        """)
        issue_types = dict(cursor.fetchall())
        
        # Route performance
        cursor.execute("""
            SELECT route, 
                   COUNT(*) as total,
                   SUM(CASE WHEN rating = 'positive' THEN 1 ELSE 0 END) as positive_count,
                   SUM(CASE WHEN rating = 'negative' THEN 1 ELSE 0 END) as negative_count
            FROM feedback 
            WHERE route IS NOT NULL
            GROUP BY route
        """)
        route_performance = cursor.fetchall()
        
        conn.close()
        
        return {
            "total": total,
            "positive": positive,
            "negative": negative,
            "satisfaction_rate": (positive / total * 100) if total > 0 else 0,
            "issue_types": issue_types,
            "route_performance": route_performance
        }
    
    def get_most_asked_questions(self, limit: int = 10) -> List[tuple]:
        """Get most frequently asked questions"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT question, COUNT(*) as count
            FROM chat_history
            GROUP BY LOWER(question)
            ORDER BY count DESC
            LIMIT ?
        """, (limit,))
        
        results = cursor.fetchall()
        conn.close()
        
        return results