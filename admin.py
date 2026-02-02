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

# ==================== TAB 1: STATS ====================
with tab1:
    st.header("📈 Genel İstatistikler")
    
    stats = db.get_feedback_stats()
    
    # Yeni format: stats["ratings"] bir dict → {"positive": N, "negative": N}
    total = stats.get("total", 0)
    ratings = stats.get("ratings", {})
    positive = ratings.get("positive", 0)
    negative = ratings.get("negative", 0)
    satisfaction = stats.get("avg_rating", 0.0)
    
    # Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Toplam Feedback", total)
    
    with col2:
        st.metric("👍 Pozitif", positive)
    
    with col3:
        st.metric("👎 Negatif", negative)
    
    with col4:
        st.metric("Memnuniyet", f"{satisfaction:.1f}%")
    
    st.markdown("---")
    
    # Rating distribution chart
    if total > 0 and ratings:
        st.subheader("📊 Rating Dağılımı")
        rating_df = pd.DataFrame([
            {"Rating": k.capitalize(), "Sayı": v}
            for k, v in ratings.items()
        ])
        st.bar_chart(rating_df.set_index("Rating"))
    
    st.markdown("---")
    
    # Route & issue type performance from all feedback
    all_feedback = db.get_all_feedback(limit=10000)
    
    if all_feedback:
        # Issue types
        issue_types = {}
        for fb in all_feedback:
            issue = fb.get("issue_type")
            if issue:
                issue_types[issue] = issue_types.get(issue, 0) + 1
        
        if issue_types:
            st.subheader("🔴 Sorun Tipleri")
            issue_df = pd.DataFrame([
                {"Sorun Tipi": k, "Sayı": v}
                for k, v in issue_types.items()
            ])
            st.bar_chart(issue_df.set_index("Sorun Tipi"))
            st.markdown("---")
        
        # Route performance
        route_stats = {}
        for fb in all_feedback:
            route = fb.get("route") or "unknown"
            if route not in route_stats:
                route_stats[route] = {"total": 0, "positive": 0, "negative": 0}
            route_stats[route]["total"] += 1
            if fb.get("rating") == "positive":
                route_stats[route]["positive"] += 1
            elif fb.get("rating") == "negative":
                route_stats[route]["negative"] += 1
        
        if route_stats:
            st.subheader("🎯 Route Performansı")
            route_data = []
            for route, data in route_stats.items():
                route_data.append({
                    "Route": route.upper(),
                    "Toplam": data["total"],
                    "Pozitif": data["positive"],
                    "Negatif": data["negative"],
                    "Başarı Oranı": f"{(data['positive'] / data['total'] * 100):.1f}%" if data["total"] > 0 else "0%"
                })
            route_df = pd.DataFrame(route_data)
            st.dataframe(route_df, use_container_width=True)

# ==================== TAB 2: NEGATIVE FEEDBACK ====================
with tab2:
    st.header("👎 Negatif Feedback")
    
    all_feedback = db.get_all_feedback(limit=200)
    negative_feedback = [fb for fb in all_feedback if fb.get("rating") == "negative"]
    
    if negative_feedback:
        for feedback in negative_feedback:
            with st.expander(
                f"❌ {feedback.get('created_at', 'N/A')} - {feedback.get('issue_type') or 'Belirtilmemiş'}"
            ):
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.markdown(f"**Soru:** {feedback.get('question', '')}")
                    answer = feedback.get('answer', '')
                    st.markdown(f"**Cevap:** {answer[:200]}{'...' if len(answer) > 200 else ''}")
                    
                    if feedback.get('comment'):
                        st.markdown(f"**Kullanıcı Yorumu:** {feedback['comment']}")
                
                with col2:
                    st.markdown(f"**Route:** {feedback.get('route') or 'N/A'}")
                    session_id = feedback.get('session_id', '')
                    st.markdown(f"**Session:** {session_id[:8]}...")
                    
                    if feedback.get('user_email'):
                        st.markdown(f"**Email:** {feedback['user_email']}")
                    
                    if feedback.get('sources'):
                        sources = feedback['sources'].split(',')
                        st.markdown(f"**Kaynaklar:** {len(sources)} döküman")
    else:
        st.info("Henüz negatif feedback yok! 🎉")

# ==================== TAB 3: ALL FEEDBACK ====================
with tab3:
    st.header("💬 Tüm Feedback")
    
    all_feedback = db.get_all_feedback(limit=200)
    
    if all_feedback:
        df = pd.DataFrame(all_feedback)
        
        # Filter options
        col1, col2 = st.columns(2)
        
        with col1:
            rating_filter = st.selectbox(
                "Rating Filtrele",
                ["Hepsi", "Pozitif", "Negatif"]
            )
        
        with col2:
            routes = df['route'].dropna().unique().tolist() if 'route' in df.columns else []
            route_filter = st.selectbox(
                "Route Filtrele",
                ["Hepsi"] + routes
            )
        
        # Apply filters
        filtered_df = df.copy()
        
        if rating_filter == "Pozitif":
            filtered_df = filtered_df[filtered_df['rating'] == 'positive']
        elif rating_filter == "Negatif":
            filtered_df = filtered_df[filtered_df['rating'] == 'negative']
        
        if route_filter != "Hepsi":
            filtered_df = filtered_df[filtered_df['route'] == route_filter]
        
        # Display columns
        display_cols = [c for c in ['created_at', 'rating', 'route', 'issue_type', 'question', 'answer', 'comment'] if c in filtered_df.columns]
        
        st.dataframe(filtered_df[display_cols], use_container_width=True)
        
        # Download
        csv = filtered_df.to_csv(index=False)
        st.download_button(
            label="📥 CSV İndir",
            data=csv,
            file_name=f"feedback_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
    else:
        st.info("Henüz feedback yok!")

# ==================== TAB 4: MOST ASKED ====================
with tab4:
    st.header("❓ En Çok Sorulan Sorular")
    
    all_feedback = db.get_all_feedback(limit=10000)
    
    if all_feedback:
        question_counts = {}
        for fb in all_feedback:
            q = fb.get("question", "")
            if q:
                question_counts[q] = question_counts.get(q, 0) + 1
        
        sorted_questions = sorted(question_counts.items(), key=lambda x: x[1], reverse=True)[:20]
        
        if sorted_questions:
            for idx, (question, count) in enumerate(sorted_questions, 1):
                st.markdown(f"**{idx}.** {question} `({count} kez)`")
        else:
            st.info("Henüz soru sorulmamış!")
    else:
        st.info("Henüz feedback yok!")

# Footer
st.markdown("---")
st.markdown("🔒 **Admin Panel** - Sadece yetkili kullanıcılar için")