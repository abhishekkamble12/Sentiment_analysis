import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
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
# CUSTOM PROFESSIONAL CSS
# ================================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    .stApp {
        background: #f8fafc;
    }
    .main-title {
        font-size: 2.8rem;
        font-weight: 700;
        background: linear-gradient(90deg, #1e40af, #3b82f6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .subtitle {
        text-align: center;
        color: #64748b;
        font-size: 1.25rem;
    }
    .metric-card {
        background: white;
        padding: 1.2rem;
        border-radius: 16px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.08);
        border: 1px solid #e2e8f0;
    }
    .stMetric {
        background: white;
        padding: 1rem;
        border-radius: 12px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.06);
    }
</style>
""", unsafe_allow_html=True)

# ================================================================
# SESSION STATE & MOCK DATA
# ================================================================
if 'current_page' not in st.session_state:
    st.session_state.current_page = "📊 Dashboard Overview"

# Generate mock data
np.random.seed(42)
dates = [datetime(2026, 3, 1) + timedelta(days=i) for i in range(30)]

mock_data = pd.DataFrame({
    'date': dates,
    'positive': np.random.randint(45, 85, 30),
    'negative': np.random.randint(8, 35, 30),
    'neutral': np.random.randint(10, 25, 30)
})
mock_data['total'] = mock_data['positive'] + mock_data['negative'] + mock_data['neutral']
mock_data['approval_score'] = (mock_data['positive'] / mock_data['total'] * 100).round(1)

# Overall stats
total_comments = 12487
overall_approval = 76.4
top_keyword = "Healthcare"
processing_status = "98.2%"

# Top keywords (mock TF-IDF)
top_keywords = {
    'Healthcare': 1240,
    'Education': 980,
    'Infrastructure': 875,
    'Water Supply': 720,
    'Roads': 685,
    'Transparency': 540,
    'Corruption': 490,
    'Employment': 465,
    'Pension': 410,
    'Digital Services': 385
}

# ================================================================
# SIDEBAR NAVIGATION
# ================================================================
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3209/3209994.png", width=80)
    st.title("E-Consultation")
    st.markdown("**Public Feedback Intelligence Platform**")
    
    st.markdown("---")
    
    page = st.radio(
        "Navigation",
        options=["📊 Dashboard Overview", "💬 Feedback Explorer", "🧪 AI Sandbox"],
        label_visibility="collapsed"
    )
    st.session_state.current_page = page
    
    st.markdown("---")
    st.subheader("🔍 Filters")
    
    st.date_input("Date Range", value=(datetime(2026,3,1), datetime(2026,3,30)), key="date_range")
    
    sentiment_filter = st.multiselect(
        "Sentiment Filter",
        options=["Positive", "Negative", "Neutral"],
        default=["Positive", "Negative", "Neutral"]
    )
    
    department = st.selectbox(
        "Department",
        ["All Departments", "Health", "Education", "Infrastructure", "Water", "Transport"]
    )
    
    st.info("✅ Live Model: RoBERTa + TF-IDF")

# ================================================================
# MAIN TITLE
# ================================================================
st.markdown("<h1 class='main-title'>E-Consultation Sentiment Analysis</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Real-time Public Feedback Intelligence for Government Decision Making</p>", unsafe_allow_html=True)

# ================================================================
# DASHBOARD OVERVIEW
# ================================================================
if st.session_state.current_page == "📊 Dashboard Overview":

    # Top Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Total Comments Analyzed",
            value=f"{total_comments:,}",
            delta="+428 today",
            delta_color="normal"
        )
    
    with col2:
        st.metric(
            label="Overall Approval Score",
            value=f"{overall_approval}%",
            delta="+2.3%",
            delta_color="normal"
        )
    
    with col3:
        st.metric(
            label="Top Keyword",
            value=top_keyword,
            delta="Trending"
        )
    
    with col4:
        st.metric(
            label="Processing Status",
            value=processing_status,
            delta="Live"
        )

    st.markdown("---")

    # Charts Row
    col_left, col_right = st.columns([1, 1])

    # 1. Sentiment Donut Chart
    with col_left:
        st.subheader("Sentiment Distribution")
        sentiment_values = [68, 22, 10]
        labels = ['Positive', 'Negative', 'Neutral']
        colors = ['#22c55e', '#ef4444', '#64748b']
        
        fig_donut = go.Figure(data=[go.Pie(
            labels=labels,
            values=sentiment_values,
            hole=0.65,
            marker=dict(colors=colors),
            textinfo='percent+label',
            textfont=dict(size=16)
        )])
        fig_donut.update_layout(
            height=420,
            showlegend=False,
            margin=dict(t=40, b=20, l=20, r=20)
        )
        st.plotly_chart(fig_donut, use_container_width=True)

    # 2. Top Keywords Horizontal Bar
    with col_right:
        st.subheader("Top 10 Keywords (TF-IDF)")
        keywords_df = pd.DataFrame({
            'Keyword': list(top_keywords.keys()),
            'Frequency': list(top_keywords.values())
        })
        
        fig_bar = px.bar(
            keywords_df,
            x='Frequency',
            y='Keyword',
            orientation='h',
            color='Frequency',
            color_continuous_scale='Blues',
            text='Frequency'
        )
        fig_bar.update_layout(
            height=420,
            yaxis=dict(autorange="reversed"),
            xaxis_title="Frequency Score",
            margin=dict(t=40, b=20, l=20, r=20)
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    st.markdown("---")

    # Trend Analysis
    st.subheader("30-Day Sentiment Trend")
    
    fig_trend = go.Figure()
    
    fig_trend.add_trace(go.Scatter(
        x=mock_data['date'], y=mock_data['positive'],
        mode='lines+markers', name='Positive', line=dict(color='#22c55e', width=4)
    ))
    fig_trend.add_trace(go.Scatter(
        x=mock_data['date'], y=mock_data['negative'],
        mode='lines+markers', name='Negative', line=dict(color='#ef4444', width=4)
    ))
    fig_trend.add_trace(go.Scatter(
        x=mock_data['date'], y=mock_data['neutral'],
        mode='lines+markers', name='Neutral', line=dict(color='#64748b', width=4)
    ))
    
    fig_trend.update_layout(
        height=500,
        xaxis_title="Date",
        yaxis_title="Number of Comments",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
        hovermode="x unified"
    )
    
    st.plotly_chart(fig_trend, use_container_width=True)

# Placeholder for other pages
elif st.session_state.current_page == "💬 Feedback Explorer":
    st.info("💬 Feedback Explorer Page - Coming in next version (with raw comments table, search, and filtering)")

elif st.session_state.current_page == "🧪 AI Sandbox":
    st.info("🧪 AI Sandbox - Test custom text here (will be implemented next)")

# Footer
st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: #64748b; font-size: 0.95rem;'>"
    "E-Consultation Sentiment Analysis Platform • Government of India / State Portal • "
    "Built with ❤️ using Streamlit + Plotly"
    "</p>",
    unsafe_allow_html=True
)
