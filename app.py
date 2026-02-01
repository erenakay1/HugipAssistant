"""
HUGİP Assistant - Streamlit Chat Interface
AI-powered chatbot for Haliç University Entrepreneurship Club
"""
import streamlit as st
import sys
from pathlib import Path
import uuid
import time

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv
load_dotenv()

from src.graph import build_graph, GraphState
from src.database.feedback_db import FeedbackDB

# Page config
st.set_page_config(
    page_title="HUGİP Asistan",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS - İyileştirilmiş Okunabilirlik
st.markdown("""
<style>
    /* Ana Chat Container */
    .chat-message {
        padding: 1.2rem;
        border-radius: 0.8rem;
        margin-bottom: 1.2rem;
        display: flex;
        flex-direction: column;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    /* Kullanıcı Mesajları - Mavi Ton */
    .chat-message.user {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        margin-left: 15%;
        border: 2px solid rgba(255,255,255,0.2);
    }
    
    /* Asistan Mesajları - Beyaz/Gri */
    .chat-message.assistant {
        background-color: #ffffff;
        color: #1a1a1a;
        margin-right: 15%;
        border: 2px solid #e0e0e0;
    }
    
    /* Mesaj İçeriği */
    .chat-message .message-content {
        margin-bottom: 0.8rem;
        font-size: 1rem;
        line-height: 1.6;
        font-weight: 500;
    }
    
    /* Kullanıcı mesajında yazı rengi */
    .chat-message.user .message-content {
        color: white;
    }
    
    /* Asistan mesajında yazı rengi */
    .chat-message.assistant .message-content {
        color: #2c3e50;
    }
    
    /* Meta Bilgiler */
    .chat-message .message-meta {
        font-size: 0.85rem;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    
    .chat-message.user .message-meta {
        color: rgba(255,255,255,0.9);
    }
    
    .chat-message.assistant .message-meta {
        color: #7f8c8d;
    }
    
    /* Kaynak Badge'leri */
    .source-badge {
        display: inline-block;
        background-color: #2196F3;
        color: white;
        padding: 0.3rem 0.6rem;
        border-radius: 0.4rem;
        font-size: 0.75rem;
        margin-right: 0.4rem;
        margin-top: 0.4rem;
        font-weight: 600;
        box-shadow: 0 1px 3px rgba(0,0,0,0.2);
    }
    
    /* Route Badge'leri */
    .route-badge {
        display: inline-block;
        padding: 0.3rem 0.7rem;
        border-radius: 0.4rem;
        font-size: 0.75rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .route-rag { 
        background-color: #10b981; 
        color: white;
        box-shadow: 0 2px 4px rgba(16,185,129,0.3);
    }
    
    .route-web { 
        background-color: #f59e0b; 
        color: white;
        box-shadow: 0 2px 4px rgba(245,158,11,0.3);
    }
    
    .route-direct { 
        background-color: #6b7280; 
        color: white;
        box-shadow: 0 2px 4px rgba(107,114,128,0.3);
    }
    
    /* Feedback Bölümü */
    .feedback-section {
        margin-top: 0.8rem;
        padding-top: 0.8rem;
        border-top: 2px solid #e5e7eb;
    }
    
    /* Sidebar Butonları */
    .stButton button {
        width: 100%;
        font-weight: 600;
    }
    
    /* Ana Başlık */
    h1 {
        color: #1a202c;
        font-weight: 700;
    }
    
    /* Markdown İçeriği */
    .chat-message.assistant ul, 
    .chat-message.assistant ol {
        color: #2c3e50;
        margin-left: 1.5rem;
    }
    
    .chat-message.assistant li {
        margin-bottom: 0.5rem;
    }
    
    /* Link Renkleri */
    .chat-message.assistant a {
        color: #3b82f6;
        font-weight: 600;
    }
    
    .chat-message.user a {
        color: #fbbf24;
        font-weight: 600;
    }
    
    /* MOBILE RESPONSIVE */
    @media (max-width: 768px) {
        .chat-message.user {
            margin-left: 5%;
            margin-right: 0;
        }
        
        .chat-message.assistant {
            margin-right: 5%;
            margin-left: 0;
        }
        
        .chat-message {
            padding: 1rem;
        }
        
        .chat-message .message-content {
            font-size: 0.95rem;
        }
        
        .source-badge, .route-badge {
            font-size: 0.7rem;
            padding: 0.25rem 0.5rem;
        }
    }
    
    /* TYPING ANIMATION */
    @keyframes typing {
        from { opacity: 0; transform: translateY(5px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .typing-animation {
        animation: typing 0.3s ease-out;
    }
    
    /* Typing dots animation */
    .typing-indicator {
        display: inline-block;
        padding: 0.5rem 1rem;
        background-color: #e5e7eb;
        border-radius: 1rem;
        margin-bottom: 1rem;
    }
    
    .typing-indicator span {
        display: inline-block;
        width: 8px;
        height: 8px;
        margin: 0 2px;
        background-color: #9ca3af;
        border-radius: 50%;
        animation: bounce 1.4s infinite ease-in-out;
    }
    
    .typing-indicator span:nth-child(1) {
        animation-delay: -0.32s;
    }
    
    .typing-indicator span:nth-child(2) {
        animation-delay: -0.16s;
    }
    
    @keyframes bounce {
        0%, 80%, 100% { 
            transform: scale(0.8);
            opacity: 0.5;
        }
        40% { 
            transform: scale(1.2);
            opacity: 1;
        }
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = []

if "graph" not in st.session_state:
    st.session_state.graph = build_graph()

if "db" not in st.session_state:
    st.session_state.db = FeedbackDB()

if "feedback_given" not in st.session_state:
    st.session_state.feedback_given = set()

# Sidebar
with st.sidebar:
    st.title("🎓 HUGİP Asistan")
    st.markdown("---")
    
    st.markdown("### 💡 Örnek Sorular")
    example_questions = [
        "FESTUP nedir?",
        "Social Media Talks'ta kimler konuşacak?",
        "Kulübe nasıl üye olabilirim?",
        "Yönetim kurulu kimlerden oluşur?",
        "Dış İlişkiler ekibi ne yapar?",
        "DigitalMAG hakkında bilgi ver",
    ]
    
    for question in example_questions:
        if st.button(question, key=f"example_{question}"):
            st.session_state.user_input = question
    
    st.markdown("---")
    
    # New chat button
    if st.button("🔄 Yeni Sohbet", type="primary"):
        st.session_state.session_id = str(uuid.uuid4())
        st.session_state.messages = []
        st.session_state.feedback_given = set()
        st.rerun()
    
    st.markdown("---")
    st.markdown("### 📊 Sohbet Bilgileri")
    st.info(f"**Mesaj Sayısı:** {len(st.session_state.messages)}")
    
    # Session ID (for debug)
    with st.expander("🔍 Session ID"):
        st.code(st.session_state.session_id, language="text")

# Main chat area
st.title("💬 HUGİP Kulüp Asistanı")
st.markdown("Haliç Üniversitesi Girişimcilik ve Pazarlama Kulübü hakkında sorularınızı sorabilirsiniz!")

# Display chat messages
chat_container = st.container()

with chat_container:
    for idx, message in enumerate(st.session_state.messages):
        role = message["role"]
        content = message["content"]
        
        if role == "user":
            st.markdown(f"""
            <div class="chat-message user">
                <div class="message-content"><strong>Siz:</strong> {content}</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            # Assistant message
            sources = message.get("sources", [])
            route = message.get("route", "unknown")
            
            # Route badge
            route_emoji = {"rag": "📚", "web_search": "🌐", "direct": "💬"}
            route_class = {"rag": "route-rag", "web_search": "route-web", "direct": "route-direct"}
            route_text = {"rag": "RAG", "web_search": "Web", "direct": "Direct"}
            
            sources_html = ""
            if sources:
                unique_sources = list(set(sources))
                sources_html = "<div style='margin-top: 0.5rem;'>"
                for source in unique_sources[:3]:  # Max 3 sources
                    source_name = source.split('/')[-1].split('\\')[-1][:30]
                    sources_html += f'<span class="source-badge">📄 {source_name}</span>'
                sources_html += "</div>"
            
            st.markdown(f"""
            <div class="chat-message assistant">
                <div class="message-content"><strong>Asistan:</strong> {content}</div>
                <div class="message-meta">
                    <span class="route-badge {route_class.get(route, 'route-direct')}">
                        {route_emoji.get(route, '💬')} {route_text.get(route, 'Direct')}
                    </span>
                </div>
                {sources_html}
            </div>
            """, unsafe_allow_html=True)
            
            # Feedback section
            feedback_key = f"feedback_{idx}"
            if feedback_key not in st.session_state.feedback_given:
                col1, col2, col3 = st.columns([1, 1, 8])
                
                with col1:
                    if st.button("👍", key=f"thumbs_up_{idx}"):
                        # Save positive feedback
                        st.session_state.db.add_feedback(
                            session_id=st.session_state.session_id,
                            question=st.session_state.messages[idx-1]["content"],
                            answer=content,
                            rating="positive",
                            route=route,
                            sources=",".join(sources) if sources else None
                        )
                        st.session_state.feedback_given.add(feedback_key)
                        st.success("Teşekkürler! 👍")
                        time.sleep(1)
                        st.rerun()
                
                with col2:
                    if st.button("👎", key=f"thumbs_down_{idx}"):
                        st.session_state[f"show_feedback_form_{idx}"] = True
                        st.rerun()
                
                # Feedback form (if thumbs down clicked)
                if st.session_state.get(f"show_feedback_form_{idx}", False):
                    with st.form(key=f"feedback_form_{idx}"):
                        st.markdown("#### ❌ Cevap nasıl iyileştirilebilir?")
                        
                        issue_type = st.selectbox(
                            "Sorun nedir?",
                            [
                                "Cevap alakasız",
                                "Bilgi eksik",
                                "Bilgi yanlış",
                                "Kaynak bulunamadı",
                                "Cevap anlaşılır değil",
                                "Diğer"
                            ],
                            key=f"issue_type_{idx}"
                        )
                        
                        comment = st.text_area(
                            "Detaylı açıklama (opsiyonel)",
                            placeholder="Ne bekliyordunuz? Nasıl daha iyi olabilirdi?",
                            key=f"comment_{idx}"
                        )
                        
                        user_email = st.text_input(
                            "Email (opsiyonel - geri dönüş için)",
                            placeholder="ornek@ogrenci.halic.edu.tr",
                            key=f"email_{idx}"
                        )
                        
                        submitted = st.form_submit_button("📤 Gönder")
                        
                        if submitted:
                            # Save negative feedback
                            st.session_state.db.add_feedback(
                                session_id=st.session_state.session_id,
                                question=st.session_state.messages[idx-1]["content"],
                                answer=content,
                                rating="negative",
                                route=route,
                                sources=",".join(sources) if sources else None,
                                issue_type=issue_type,
                                comment=comment if comment else None,
                                user_email=user_email if user_email else None
                            )
                            st.session_state.feedback_given.add(feedback_key)
                            st.session_state[f"show_feedback_form_{idx}"] = False
                            st.success("Geri bildiriminiz kaydedildi! Teşekkürler! 🙏")
                            time.sleep(1)
                            st.rerun()

# Chat input
user_input = st.chat_input("Mesajınızı yazın...")

# Handle example question click
if "user_input" in st.session_state:
    user_input = st.session_state.user_input
    del st.session_state.user_input

if user_input:
    # Add user message
    st.session_state.messages.append({
        "role": "user",
        "content": user_input
    })
    
    # Rerun to show user message immediately
    st.rerun()

# Check if we need to generate response (last message is user)
if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
    # Show typing indicator
    with st.chat_message("assistant"):
        typing_placeholder = st.empty()
        typing_placeholder.markdown("""
        <div class="typing-indicator">
            <span></span>
            <span></span>
            <span></span>
        </div>
        """, unsafe_allow_html=True)
        
        start_time = time.time()
        
        # Get the last user message
        last_user_message = st.session_state.messages[-1]["content"]
        
        # Invoke graph
        result = st.session_state.graph.invoke({
            "question": last_user_message,
            "generation": "",
            "documents": [],
            "decision": "",
            "web_results": [],
            "iterations": 0,
            "session_id": st.session_state.session_id
        })
        
        response_time = time.time() - start_time
        
        # Clear typing indicator
        typing_placeholder.empty()
        
        # Extract sources
        sources = []
        if result.get("documents"):
            sources = [doc.metadata.get("source", "Unknown") for doc in result["documents"]]
        
        # Add assistant message
        st.session_state.messages.append({
            "role": "assistant",
            "content": result["generation"],
            "sources": sources,
            "route": result["decision"]
        })
        
        # Log to database
        st.session_state.db.log_chat(
            session_id=st.session_state.session_id,
            question=last_user_message,
            answer=result["generation"],
            route=result["decision"],
            sources=",".join(sources) if sources else None,
            response_time=response_time
        )
        
        # Display response with typing effect
        response_container = st.empty()
        full_response = result["generation"]
        displayed_text = ""
        
        # Simulate typing (split by words for smoother effect)
        words = full_response.split()
        for i, word in enumerate(words):
            displayed_text += word + " "
            response_container.markdown(f"""
            <div class="typing-animation">
                {displayed_text}
            </div>
            """, unsafe_allow_html=True)
            time.sleep(0.03)  # 30ms delay between words
        
        # Final rerun to show complete message with sources/badges
        time.sleep(0.5)
    
    st.rerun()

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; font-size: 0.9rem;'>
    <p>🎓 Haliç Üniversitesi Girişimcilik ve Pazarlama Kulübü</p>
    <p>Powered by AI | Made with ❤️ for HUGİP</p>
</div>
""", unsafe_allow_html=True)