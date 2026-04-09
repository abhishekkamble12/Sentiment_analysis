import streamlit as st
import pandas as pd
import numpy as np
import re
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
# CUSTOM CSS
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
    }
    .result-card {
        padding: 2.5rem;
        border-radius: 20px;
        text-align: center;
        margin: 1.5rem 0;
        animation: fadeIn 0.6s;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
    }
    @keyframes fadeIn {
        from {opacity: 0; transform: translateY(20px);}
        to {opacity: 1; transform: translateY(0);}
    }
</style>
""", unsafe_allow_html=True)

# ================================================================
# MOCK DATA FOR DASHBOARD
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
    st.markdown("**Public Feedback Intelligence**")
    
    st.markdown("---")
    page = st.radio(
        "Navigation",
        ["📊 Dashboard Overview", "💬 Analyze Comment", "💬 Feedback Explorer"]
    )
    
    st.markdown("---")
    st.subheader("🔍 Filters")
    st.date_input("Date Range", value=(datetime(2026,3,1).date(), datetime.now().date()))
    st.multiselect("Sentiment", ["Positive", "Negative", "Neutral"], default=["Positive", "Negative", "Neutral"])

# ================================================================
# HEADER
# ================================================================
st.markdown("<h1 class='main-title'>E-Consultation Sentiment Analysis Platform</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; color:#475569; font-size:1.2rem;'>Government Public Feedback Monitoring System</p>", unsafe_allow_html=True)

# ================================================================
# PAGE 1: DASHBOARD OVERVIEW
# ================================================================
if page == "📊 Dashboard Overview":
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Comments", "12,487", "↑ 428 today")
    with col2:
        st.metric("Overall Approval", "76.4%", "↑ 2.3%")
    with col3:
        st.metric("Top Keyword", "Healthcare", "Trending")
    with col4:
        st.metric("Model Accuracy", "98.2%", "Live")

    st.markdown("---")

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Sentiment Distribution")
        sentiment = pd.DataFrame({
            "Sentiment": ["Positive", "Negative", "Neutral"],
            "Count": [8520, 2750, 1217]
        })
        st.bar_chart(sentiment.set_index("Sentiment"), color=["#22c55e"])

    with c2:
        st.subheader("Top 10 Keywords")
        st.bar_chart(top_keywords.set_index("Keyword")["Frequency"], color=["#3b82f6"])

    st.markdown("---")
    st.subheader("30-Day Sentiment Trend")
    trend = data[['date', 'positive', 'negative', 'neutral']].set_index('date')
    st.line_chart(trend, color=["#22c55e", "#ef4444", "#64748b"])

# ================================================================
# PAGE 2: ANALYZE COMMENT (Like Original Code)
# ================================================================
elif page == "💬 Analyze Comment":
    
    st.subheader("✍️ Enter Public Feedback / Comment")
    
    text = st.text_area(
        "Write or paste the comment here:",
        placeholder="Type the citizen's feedback here...",
        height=180
    )

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        analyze_btn = st.button("🚀 Analyze Sentiment", type="primary", use_container_width=True)

    if analyze_btn:
        if text.strip() == "":
            st.warning("⚠️ Please enter some text to analyze.")
        else:
            with st.spinner("Analyzing sentiment using AI model..."):
                # Mock Analysis (Replace with real model later)
                cleaned = re.sub(r'[^a-zA-Z\s]', '', text.lower())
                time.sleep(1.2)
                
                # Simple mock logic for demo
                positive_words = ['good', 'great', 'excellent', 'love', 'best', 'happy', 'thank', 'support']
                negative_words = ['bad', 'worst', 'poor', 'terrible', 'hate', 'problem', 'issue', 'delay']
                
                pos_count = sum(1 for word in positive_words if word in cleaned)
                neg_count = sum(1 for word in negative_words if word in cleaned)
                
                if pos_count > neg_count:
                    pred = "positive"
                    conf = 75 + np.random.randint(10, 25)
                    emoji = "😊"
                    color = "linear-gradient(135deg, #22c55e, #86efac)"
                    text_color = "#14532d"
                elif neg_count > pos_count:
                    pred = "negative"
                    conf = 70 + np.random.randint(10, 25)
                    emoji = "😠"
                    color = "linear-gradient(135deg, #ef4444, #f87171)"
                    text_color = "white"
                else:
                    pred = "neutral"
                    conf = 65 + np.random.randint(10, 25)
                    emoji = "😐"
                    color = "linear-gradient(135deg, #64748b, #94a3b8)"
                    text_color = "white"

            # Result Card
            st.markdown(f"""
            <div class="result-card" style="background:{color}; color:{text_color};">
                <h1 style="font-size: 4.5rem; margin:0;">{emoji}</h1>
                <h2 style="margin:10px 0;">{pred.upper()}</h2>
                <h3>Confidence: {conf}%</h3>
                <p style="font-size:1.1rem; margin-top:15px;">
                    { "This feedback shows strong positive sentiment." if pred == "positive" else 
                      "This feedback expresses negative sentiment." if pred == "negative" else 
                      "This feedback appears neutral and balanced." }
                </p>
            </div>
            """, unsafe_allow_html=True)

            # Save to history
            if 'analysis_history' not in st.session_state:
                st.session_state.analysis_history = []
            
            st.session_state.analysis_history.append({
                "time": datetime.now().strftime("%H:%M:%S"),
                "text": text[:120] + "..." if len(text) > 120 else text,
                "sentiment": pred.capitalize(),
                "confidence": conf
            })

    # Show Recent Analyses
    if 'analysis_history' in st.session_state and st.session_state.analysis_history:
        st.markdown("---")
        st.subheader("Recent Analyses")
        for item in reversed(st.session_state.analysis_history[-5:]):
            st.info(f"**{item['time']}** | **{item['sentiment']}** ({item['confidence']}%) — {item['text']}")

# ================================================================
# PAGE 3: FEEDBACK EXPLORER
# ================================================================
elif page == "💬 Feedback Explorer":
    st.subheader("All Public Feedback Comments")
    st.caption("Searchable list of analyzed comments")
    
    # Mock table
    sample_data = pd.DataFrame({
        "Date": ["2026-04-08", "2026-04-07", "2026-04-07", "2026-04-06"],
        "Comment": [
            "The new healthcare scheme is very helpful...",
            "Road conditions are very bad in our area.",
            "Education department is doing good work.",
            "Water supply is irregular in our village."
        ],
        "Sentiment": ["Positive", "Negative", "Positive", "Negative"],
        "Confidence": [82, 88, 79, 91]
    })
    
    st.dataframe(sample_data, use_container_width=True, hide_index=True)

# Footer
st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: #64748b;'>E-Consultation Sentiment Analysis Platform • Government Stakeholder Tool</p>",
    unsafe_allow_html=True
)
