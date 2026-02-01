"""
HUGİP Assistant - Admin Panel
View feedback, analytics, and system stats
"""
import streamlit as st
import sys
from pathlib import Path
import pandas as pd
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.database.feedback_db import FeedbackDB

# Page config
st.set_page_config(
    page_title="HUGİP Asistan - Admin Panel",
    page_icon="📊",
    layout="wide"
)

# Initialize DB
db = FeedbackDB()

# Header
st.title("📊 HUGİP Asistan - Admin Panel")
st.markdown("Feedback ve Analytics Dashboard")

# Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "📈 Genel İstatistikler",
    "👎 Negatif Feedback",
    "💬 Tüm Feedback",
    "❓ En Çok Sorulan Sorular"
])

# Tab 1: Stats
with tab1:
    st.header("📈 Genel İstatistikler")
    
    stats = db.get_feedback_stats()
    
    # Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Toplam Feedback", stats["total"])
    
    with col2:
        st.metric("👍 Pozitif", stats["positive"])
    
    with col3:
        st.metric("👎 Negatif", stats["negative"])
    
    with col4:
        satisfaction = stats["satisfaction_rate"]
        st.metric("Memnuniyet", f"{satisfaction:.1f}%")
    
    st.markdown("---")
    
    # Issue types
    if stats["issue_types"]:
        st.subheader("🔴 Sorun Tipleri")
        issue_df = pd.DataFrame([
            {"Sorun Tipi": k, "Sayı": v} 
            for k, v in stats["issue_types"].items()
        ])
        st.bar_chart(issue_df.set_index("Sorun Tipi"))
    
    st.markdown("---")
    
    # Route performance
    if stats["route_performance"]:
        st.subheader("🎯 Route Performansı")
        route_data = []
        for route, total, positive, negative in stats["route_performance"]:
            route_data.append({
                "Route": route.upper(),
                "Toplam": total,
                "Pozitif": positive,
                "Negatif": negative,
                "Başarı Oranı": f"{(positive/total*100):.1f}%" if total > 0 else "0%"
            })
        
        route_df = pd.DataFrame(route_data)
        st.dataframe(route_df, use_container_width=True)

# Tab 2: Negative Feedback
with tab2:
    st.header("👎 Negatif Feedback")
    
    negative_feedback = db.get_negative_feedback(limit=100)
    
    if negative_feedback:
        for feedback in negative_feedback:
            with st.expander(
                f"❌ {feedback['created_at']} - {feedback['issue_type'] or 'Belirtilmemiş'}"
            ):
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.markdown(f"**Soru:** {feedback['question']}")
                    st.markdown(f"**Cevap:** {feedback['answer'][:200]}...")
                    
                    if feedback['comment']:
                        st.markdown(f"**Kullanıcı Yorumu:** {feedback['comment']}")
                
                with col2:
                    st.markdown(f"**Route:** {feedback['route'] or 'N/A'}")
                    st.markdown(f"**Session:** {feedback['session_id'][:8]}...")
                    
                    if feedback['user_email']:
                        st.markdown(f"**Email:** {feedback['user_email']}")
                    
                    if feedback['sources']:
                        sources = feedback['sources'].split(',')
                        st.markdown(f"**Kaynaklar:** {len(sources)} döküman")
    else:
        st.info("Henüz negatif feedback yok! 🎉")

# Tab 3: All Feedback
with tab3:
    st.header("💬 Tüm Feedback")
    
    all_feedback = db.get_all_feedback(limit=200)
    
    if all_feedback:
        # Convert to DataFrame
        df = pd.DataFrame(all_feedback)
        
        # Filter options
        col1, col2 = st.columns(2)
        
        with col1:
            rating_filter = st.selectbox(
                "Rating Filtrele",
                ["Hepsi", "Pozitif", "Negatif"]
            )
        
        with col2:
            route_filter = st.selectbox(
                "Route Filtrele",
                ["Hepsi"] + list(df['route'].dropna().unique())
            )
        
        # Apply filters
        filtered_df = df.copy()
        
        if rating_filter == "Pozitif":
            filtered_df = filtered_df[filtered_df['rating'] == 'positive']
        elif rating_filter == "Negatif":
            filtered_df = filtered_df[filtered_df['rating'] == 'negative']
        
        if route_filter != "Hepsi":
            filtered_df = filtered_df[filtered_df['route'] == route_filter]
        
        # Display table
        st.dataframe(
            filtered_df[[
                'created_at', 'rating', 'route', 'issue_type', 
                'question', 'answer', 'comment'
            ]],
            use_container_width=True
        )
        
        # Download button
        csv = filtered_df.to_csv(index=False)
        st.download_button(
            label="📥 CSV İndir",
            data=csv,
            file_name=f"feedback_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
    else:
        st.info("Henüz feedback yok!")

# Tab 4: Most Asked Questions
with tab4:
    st.header("❓ En Çok Sorulan Sorular")
    
    most_asked = db.get_most_asked_questions(limit=20)
    
    if most_asked:
        for idx, (question, count) in enumerate(most_asked, 1):
            st.markdown(f"**{idx}.** {question} ({count} kez)")
    else:
        st.info("Henüz soru sorulmamış!")

# Footer
st.markdown("---")
st.markdown("🔒 **Admin Panel** - Sadece yetkili kullanıcılar için")