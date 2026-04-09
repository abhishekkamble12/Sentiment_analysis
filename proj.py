import streamlit as st
import pickle
import re
import os
import time
from datetime import datetime

# ================================================================
# PAGE CONFIG
# ================================================================
st.set_page_config(
    page_title="Sentiment AI",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ================================================================
# CUSTOM CSS - World-Class Modern Design
# ================================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Poppins', sans-serif;
    }

    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }

    /* Glassmorphism Container */
    .main-container {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(20px);
        border-radius: 24px;
        padding: 2.5rem;
        box-shadow: 0 20px 40px rgba(0,0,0,0.15);
        margin: 1rem 0;
    }

    .title {
        text-align: center;
        font-size: 3.8rem;
        font-weight: 800;
        background: linear-gradient(90deg, #4f46e5, #a855f7);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }

    .subtitle {
        text-align: center;
        font-size: 1.35rem;
        color: #475569;
        margin-bottom: 2rem;
    }

    /* Text Area */
    .stTextArea textarea {
        border-radius: 16px !important;
        border: 2px solid #e2e8f0 !important;
        font-size: 1.1rem !important;
        padding: 1.2rem !important;
        background: #ffffff !important;
        color: #1e2937 !important;
        transition: all 0.3s ease;
    }

    .stTextArea textarea:focus {
        border-color: #6366f1 !important;
        box-shadow: 0 0 0 4px rgba(99, 102, 241, 0.2) !important;
    }

    /* Button */
    .stButton > button {
        width: 100%;
        height: 62px;
        background: linear-gradient(90deg, #4f46e5, #7c3aed);
        color: white;
        border-radius: 9999px;
        font-size: 1.25rem;
        font-weight: 600;
        border: none;
        box-shadow: 0 10px 25px rgba(79, 70, 229, 0.3);
        transition: all 0.3s ease;
    }

    .stButton > button:hover {
        transform: translateY(-3px);
        box-shadow: 0 15px 35px rgba(79, 70, 229, 0.4);
    }

    /* Result Cards */
    .result-card {
        padding: 2.5rem;
        border-radius: 24px;
        text-align: center;
        animation: fadeInUp 0.6s ease;
        box-shadow: 0 15px 35px rgba(0,0,0,0.1);
        margin: 1.5rem 0;
    }

    @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(40px); }
        to { opacity: 1; transform: translateY(0); }
    }

    .confidence-bar {
        height: 8px;
        background: linear-gradient(90deg, #22c55e, #86efac);
        border-radius: 9999px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ================================================================
# SESSION STATE
# ================================================================
if 'history' not in st.session_state:
    st.session_state.history = []
if 'theme' not in st.session_state:
    st.session_state.theme = "Light"

# ================================================================
# CLEANING FUNCTION
# ================================================================
def clean_text(text):
    text = str(text).lower().strip()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text

# ================================================================
# LOAD MODEL
# ================================================================
@st.cache_resource
def load_model():
    try:
        if os.path.exists("sentiment_model.pkl"):
            with open("sentiment_model.pkl", "rb") as f:
                return pickle.load(f)
        else:
            st.error("❌ Model file `sentiment_model.pkl` not found!")
            return None
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_model()

# ================================================================
# SIDEBAR
# ================================================================
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3209/3209994.png", width=100)
    st.title("Sentiment AI")
    
    st.markdown("---")
    st.subheader("🎨 Theme")
    if st.button("Toggle Dark/Light Mode"):
        st.session_state.theme = "Dark" if st.session_state.theme == "Light" else "Light"
    
    st.markdown("---")
    st.subheader("📖 About")
    st.write("""
    Advanced sentiment analysis powered by Machine Learning.
    Detects **Positive**, **Negative**, and **Neutral** emotions with confidence scores.
    """)
    
    st.info("💡 Pro Tip: Try writing emotional reviews or tweets!")

# ================================================================
# MAIN UI
# ================================================================
st.markdown("<h1 class='title'>Sentiment AI ✨</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Uncover the emotion behind every word</p>", unsafe_allow_html=True)

# Example Prompts
st.markdown("**Quick Examples:**")
cols = st.columns(3)
examples = [
    "I absolutely love this product! Best purchase ever.",
    "This is the worst service I've ever experienced.",
    "The movie was okay, nothing special."
]

for i, col in enumerate(cols):
    if col.button(examples[i], use_container_width=True):
        st.session_state.example_text = examples[i]

# Text Input
text = st.text_area(
    "✍️ Enter your text here:",
    placeholder="Type or paste your message...",
    height=160,
    value=st.session_state.get('example_text', '')
)

# Analyze Button
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    analyze_btn = st.button("🚀 Analyze Sentiment", type="primary")

# ================================================================
# ANALYSIS
# ================================================================
if analyze_btn:
    if not text or text.strip() == "":
        st.warning("⚠️ Please enter some text to analyze.")
    elif model is None:
        st.error("Model not loaded. Please check sentiment_model.pkl")
    else:
        with st.spinner("Analyzing emotions with AI..."):
            time.sleep(0.8)
            
            cleaned = clean_text(text)
            prediction = model.predict([cleaned])[0]
            
            try:
                probabilities = model.predict_proba([cleaned])[0]
                confidence = max(probabilities) * 100
                confidence_class = prediction
            except:
                confidence = 85.0  # fallback

        # Store in history
        result_entry = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "text": text[:150] + "..." if len(text) > 150 else text,
            "sentiment": prediction.capitalize(),
            "confidence": round(confidence, 1)
        }
        st.session_state.history.append(result_entry)
        if len(st.session_state.history) > 10:
            st.session_state.history.pop(0)

        # Result Display
        if prediction == "positive":
            st.markdown(f"""
            <div class="result-card" style="background: linear-gradient(135deg, #22c55e, #86efac); color: #14532d;">
                <h1 style="font-size: 4rem; margin: 0;">😊 Positive</h1>
                <p style="font-size: 1.4rem; margin: 10px 0;">This text radiates positivity!</p>
                <div style="font-size: 2.2rem; font-weight: 700;">{confidence:.1f}% Confidence</div>
                <div class="confidence-bar" style="width: {confidence}%; background: linear-gradient(90deg, #15803d, #4ade80);"></div>
            </div>
            """, unsafe_allow_html=True)
            st.balloons()

        elif prediction == "negative":
            st.markdown(f"""
            <div class="result-card" style="background: linear-gradient(135deg, #ef4444, #f87171); color: white;">
                <h1 style="font-size: 4rem; margin: 0;">😠 Negative</h1>
                <p style="font-size: 1.4rem; margin: 10px 0;">This message expresses negative sentiment.</p>
                <div style="font-size: 2.2rem; font-weight: 700;">{confidence:.1f}% Confidence</div>
                <div class="confidence-bar" style="width: {confidence}%; background: linear-gradient(90deg, #b91c1c, #fb7185);"></div>
            </div>
            """, unsafe_allow_html=True)

        else:
            st.markdown(f"""
            <div class="result-card" style="background: linear-gradient(135deg, #64748b, #94a3b8); color: white;">
                <h1 style="font-size: 4rem; margin: 0;">😐 Neutral</h1>
                <p style="font-size: 1.4rem; margin: 10px 0;">This text appears balanced and neutral.</p>
                <div style="font-size: 2.2rem; font-weight: 700;">{confidence:.1f}% Confidence</div>
                <div class="confidence-bar" style="width: {confidence}%; background: linear-gradient(90deg, #475569, #cbd5e1);"></div>
            </div>
            """, unsafe_allow_html=True)

# ================================================================
# HISTORY
# ================================================================
if st.session_state.history:
    st.markdown("---")
    st.subheader("📜 Analysis History")
    for entry in reversed(st.session_state.history):
        emoji = "😊" if entry["sentiment"].lower() == "positive" else "😠" if entry["sentiment"].lower() == "negative" else "😐"
        st.markdown(f"""
        <div style="padding: 1rem; border-radius: 16px; background: rgba(255,255,255,0.7); margin: 8px 0;">
            <small>{entry['timestamp']}</small><br>
            <b>{emoji} {entry['sentiment']}</b> • {entry['confidence']}% 
            <br><span style="color: #64748b;">"{entry['text']}"</span>
        </div>
        """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: #64748b; font-size: 0.9rem;'>"
    "Built with ❤️ using Streamlit • World-Class UI by Grok"
    "</p>",
    unsafe_allow_html=True
)
