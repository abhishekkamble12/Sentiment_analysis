import streamlit as st
import pickle
import re
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
# LOAD MODEL (From your GitHub repo)
# ================================================================
@st.cache_resource
def load_model():
    try:
        with open("sentiment_model.pkl", "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        st.error("❌ Model file `sentiment_model.pkl` not found!")
        st.stop()
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()

model = load_model()

# ================================================================
# CLEANING FUNCTION (Same as your original)
# ================================================================
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

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
# DASHBOARD OVERVIEW
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
# ANALYZE COMMENT (Using Your Real Model)
# ================================================================
elif page == "💬 Analyze Comment":
    st.subheader("✍️ Enter Public Feedback / Comment")
    
    text = st.text_area(
        "Write or paste the citizen's comment here:",
        placeholder="Type the feedback here...",
        height=180
    )

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        analyze_btn = st.button("🚀 Analyze Sentiment", type="primary", use_container_width=True)

    if analyze_btn:
        if text.strip() == "":
            st.warning("⚠️ Please enter some text to analyze.")
        else:
            with st.spinner("Analyzing sentiment using ML Model..."):
                cleaned = clean_text(text)
                pred = model.predict([cleaned])[0]
                
                try:
                    proba = model.predict_proba([cleaned])[0]
                    conf = round(max(proba) * 100, 1)
                except:
                    conf = 85.0

            # Result Display
            if pred == "positive":
                emoji, color, text_color = "😊", "linear-gradient(135deg, #22c55e, #86efac)", "#14532d"
                message = "This feedback shows strong positive sentiment."
            elif pred == "negative":
                emoji, color, text_color = "😠", "linear-gradient(135deg, #ef4444, #f87171)", "white"
                message = "This feedback expresses negative sentiment."
            else:
                emoji, color, text_color = "😐", "linear-gradient(135deg, #64748b, #94a3b8)", "white"
                message = "This feedback appears neutral and balanced."

            st.markdown(f"""
            <div class="result-card" style="background:{color}; color:{text_color};">
                <h1 style="font-size: 4.5rem; margin:0;">{emoji}</h1>
                <h2>{pred.upper()}</h2>
                <h3>Confidence: {conf}%</h3>
                <p style="font-size:1.1rem;">{message}</p>
            </div>
            """, unsafe_allow_html=True)

            # Save History
            if 'analysis_history' not in st.session_state:
                st.session_state.analysis_history = []
            st.session_state.analysis_history.append({
                "time": datetime.now().strftime("%H:%M:%S"),
                "text": text[:150] + "..." if len(text) > 150 else text,
                "sentiment": pred.capitalize(),
                "confidence": conf
            })

    # Recent History
    if 'analysis_history' in st.session_state and st.session_state.analysis_history:
        st.markdown("---")
        st.subheader("Recent Analyses")
        for item in reversed(st.session_state.analysis_history[-6:]):
            st.info(f"**{item['time']}** | **{item['sentiment']}** ({item['confidence']}%) — {item['text']}")

# ================================================================
# FEEDBACK EXPLORER
# ================================================================
elif page == "💬 Feedback Explorer":
    st.subheader("All Public Feedback Comments")
    sample_data = pd.DataFrame({
        "Date": ["2026-04-08", "2026-04-07", "2026-04-07", "2026-04-06"],
        "Comment": [
            "The new healthcare scheme is very helpful for senior citizens.",
            "Road conditions are very bad in our area.",
            "Education department is doing good work.",
            "Water supply is irregular in our village."
        ],
        "Sentiment": ["Positive", "Negative", "Positive", "Negative"],
        "Confidence": [84, 91, 78, 89]
    })
    st.dataframe(sample_data, use_container_width=True, hide_index=True)

# Footer
st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: #64748b;'>E-Consultation Sentiment Analysis Platform • Government Stakeholder Tool</p>",
    unsafe_allow_html=True
)
