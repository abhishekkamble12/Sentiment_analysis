import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# ================================================================
# PAGE CONFIG
# ================================================================
st.set_page_config(
    page_title="E-Consultation | Sentiment Analysis",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ================================================================
# CUSTOM CSS (Professional Government Look)
# ================================================================
st.markdown("""
<style>
    .main-title {
        font-size: 2.8rem;
        font-weight: 700;
        background: linear-gradient(90deg, #1e40af, #3b82f6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.3rem;
    }
    .subtitle {
        text-align: center;
        color: #64748b;
        font-size: 1.25rem;
        margin-bottom: 2rem;
    }
    .metric-container {
        background: white;
        padding: 1.2rem;
        border-radius: 16px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.08);
        border: 1px solid #e2e8f0;
    }
</style>
""", unsafe_allow_html=True)

# ================================================================
# MOCK DATA
# ================================================================
np.random.seed(42)
dates = [datetime(2026, 3, 1) + timedelta(days=i) for i in range(30)]

data = pd.DataFrame({
    'date': dates,
    'positive': np.random.randint(45, 85, 30),
    'negative': np.random.randint(8, 35, 30),
    'neutral': np.random.randint(10, 25, 30)
})
data['total'] = data['positive'] + data['negative'] + data['neutral']
data['approval'] = (data['positive'] / data['total'] * 100).round(1)

# Overall Stats
total_comments = 12487
overall_approval = 76.4
top_keyword = "Healthcare"

# Top Keywords Data
top_keywords = pd.DataFrame({
    'Keyword': ['Healthcare', 'Education', 'Infrastructure', 'Water Supply', 'Roads', 
                'Transparency', 'Employment', 'Corruption', 'Pension', 'Digital Services'],
    'Frequency': [1240, 980, 875, 720, 685, 540, 465, 490, 410, 385]
})

# ================================================================
# SIDEBAR
# ================================================================
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3209/3209994.png", width=90)
    st.title("E-Consultation")
    st.markdown("**Public Feedback Intelligence Platform**")
    
    st.markdown("---")
    
    page = st.radio(
        "Main Navigation",
        ["📊 Dashboard Overview", "💬 Feedback Explorer", "🧪 AI Sandbox"]
    )
    
    st.markdown("---")
    st.subheader("🔍 Filters")
    st.date_input("Date Range", value=(datetime(2026,3,1).date(), datetime.now().date()))
    
    st.multiselect(
        "Sentiment Filter",
        options=["Positive", "Negative", "Neutral"],
        default=["Positive", "Negative", "Neutral"]
    )
    
    st.selectbox(
        "Department",
        ["All Departments", "Health", "Education", "Infrastructure", "Water", "Transport"]
    )

# ================================================================
# HEADER
# ================================================================
st.markdown("<h1 class='main-title'>E-Consultation Sentiment Analysis</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Real-time Public Feedback Monitoring System</p>", unsafe_allow_html=True)

# ================================================================
# DASHBOARD OVERVIEW
# ================================================================
if page == "📊 Dashboard Overview":

    # Top Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Comments", f"{total_comments:,}", "↑ 428 today")
    with col2:
        st.metric("Overall Approval", f"{overall_approval}%", "↑ 2.3%")
    with col3:
        st.metric("Top Keyword", top_keyword, "Trending")
    with col4:
        st.metric("Processing Rate", "98.2%", "Live")

    st.markdown("---")

    # Sentiment Distribution & Top Keywords
    col_left, col_right = st.columns(2)

    # Sentiment Distribution (Using Simple Bar + Percentages)
    with col_left:
        st.subheader("Sentiment Distribution")
        
        sentiment_data = pd.DataFrame({
            "Sentiment": ["Positive", "Negative", "Neutral"],
            "Count": [8520, 2750, 1217],
            "Percentage": [68.2, 22.0, 9.8]
        })
        
        st.bar_chart(
            sentiment_data.set_index("Sentiment")["Count"],
            use_container_width=True,
            color=["#22c55e"]
        )
        
        # Show percentages
        for i, row in sentiment_data.iterrows():
            st.progress(row["Percentage"]/100, text=f"{row['Sentiment']}: {row['Percentage']}%")

    # Top Keywords
    with col_right:
        st.subheader("Top 10 Keywords")
        # Horizontal bar using Streamlit bar_chart
        st.bar_chart(
            top_keywords.set_index("Keyword")["Frequency"],
            use_container_width=True,
            color=["#3b82f6"]
        )

    st.markdown("---")

    # Trend Analysis
    st.subheader("30-Day Sentiment Trend")
    
    trend_df = data[['date', 'positive', 'negative', 'neutral']].set_index('date')
    st.line_chart(
        trend_df,
        use_container_width=True,
        color=["#22c55e", "#ef4444", "#64748b"]
    )

    # Summary Table
    st.subheader("Recent Summary")
    st.dataframe(
        data.tail(10)[['date', 'positive', 'negative', 'neutral', 'approval']],
        use_container_width=True,
        hide_index=True
    )

# Other Pages (Placeholders)
elif page == "💬 Feedback Explorer":
    st.info("💬 **Feedback Explorer** page is under development.\n\nIt will contain searchable comments table with filters and export option.")

elif page == "🧪 AI Sandbox":
    st.info("🧪 **AI Sandbox** coming soon.\n\nYou will be able to test live text analysis here.")

# Footer
st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: #64748b; font-size: 0.95rem;'>"
    "E-Consultation Sentiment Analysis Platform • Government Stakeholder Tool"
    "</p>",
    unsafe_allow_html=True
)
