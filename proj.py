import streamlit as st
import pickle
import re
import os
import time

# ================================================================
# 1. PAGE CONFIG
# ================================================================
st.set_page_config(
    page_title="Sentiment AI",
    page_icon="✨",
    layout="wide"
)

# ================================================================
# 2. CUSTOM THEME + CSS  (FIXED TEXT VISIBILITY)
# ================================================================
st.markdown("""
<style>

/* Global Font */
html, body, [class*="css"] {
    font-family: 'Poppins', sans-serif;
}

/* Background */
.stApp {
    background: linear-gradient(135deg, #e0e7ff 0%, #fdf2f8 100%);
}

/* Title */
.title {
    text-align: center;
    font-size: 55px !important;
    color: #2d3436;
    font-weight: 900;
    margin-top: -15px;
    text-shadow: 2px 2px 7px rgba(0,0,0,0.15);
}

/* Subtitle */
.subtitle {
    text-align: center;
    font-size: 20px;
    color: #636e72;
    margin-bottom: 25px;
}

/* TEXTAREA FIX: TEXT NOW VISIBLE */
.stTextArea textarea {
    background: #ffffff !important;
    padding: 18px;
    border-radius: 12px;
    border: 2px solid #b2bec3 !important;
    font-size: 17px;
    color: #000 !important;          /* <-- TEXT VISIBLE FIX */
    caret-color: #000 !important;     /* cursor visible */
    transition: 0.25s ease;
}

.stTextArea textarea:focus {
    border-color: #a29bfe !important;
    box-shadow: 0 0 0 4px rgba(162,155,254,0.3) !important;
}

/* Center Button */
.stButton > button {
    width: 100%;
    background: linear-gradient(90deg, #6c5ce7, #a29bfe);
    border-radius: 30px;
    padding: 12px;
    font-size: 20px;
    font-weight: 600;
    color: white;
    border: none;
    transition: 0.3s ease;
}

.stButton > button:hover {
    transform: scale(1.05);
    box-shadow: 0px 8px 18px rgba(108, 92, 231, 0.4);
}

/* Result Card Animation */
.result-card {
    padding: 28px;
    border-radius: 18px;
    text-align: center;
    animation: fadeIn 0.6s ease-in-out;
}

@keyframes fadeIn {
  from {opacity:0; transform: translateY(25px);}
  to {opacity:1; transform: translateY(0);}
}

</style>
""", unsafe_allow_html=True)

# ================================================================
# 3. CLEANING FUNCTION
# ================================================================
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    return text

# ================================================================
# 4. LOAD MODEL
# ================================================================
@st.cache_resource
def load_model():
    if os.path.exists("sentiment_model.pkl"):
        with open("sentiment_model.pkl", "rb") as f:
            return pickle.load(f)
    return None

model = load_model()

# ================================================================
# 5. SIDEBAR
# ================================================================
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3209/3209994.png", width=120)
    st.header("📘 About Sentiment AI")
    st.write("""
    This app uses a Machine  
    Learning model to classify your text as:

    ✔ Positive  
    ✔ Negative  
    ✔ Neutral  
    """)
    st.info("Developed with ❤️ using Streamlit")

# ================================================================
# 6. MAIN UI
# ================================================================
st.markdown("<h1 class='title'>Sentiment AI ✨</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Analyze emotions hidden behind your text. Fast and accurate.</p>", unsafe_allow_html=True)

# If model missing
if model is None:
    st.error("⚠️ Model not found! Upload `sentiment_model.pkl` in the same folder.")
    st.stop()

# Text Input
text = st.text_area("✍️ Write something here:", placeholder="Type your message...", height=140)

# Center the button
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    run = st.button("🚀 Analyze Now")

# ================================================================
# 7. PROCESSING
# ================================================================
if run:
    if text.strip() == "":
        st.warning("⚠️ Please enter some text before analyzing.")
    else:
        with st.spinner("Analyzing sentiment... 🔍"):
            time.sleep(0.7)
            cleaned = clean_text(text)
            pred = model.predict([cleaned])[0]
            try:
                conf = max(model.predict_proba([cleaned])[0]) * 100
            except:
                conf = 0

        # ------------------------------ RESULT UI ------------------------------
        if pred == "positive":
            st.markdown(f"""
            <div class="result-card" style="background:linear-gradient(135deg,#b2ffda,#6effb3);">
                <h1>😊 Positive</h1>
                <p><b>Confidence:</b> {conf:.1f}%</p>
                <p>This message spreads positivity!</p>
            </div>
            """, unsafe_allow_html=True)
            st.balloons()

        elif pred == "negative":
            st.markdown(f"""
            <div class="result-card" style="background:linear-gradient(135deg,#ffb199,#ff0844); color:white;">
                <h1>😡 Negative</h1>
                <p><b>Confidence:</b> {conf:.1f}%</p>
                <p>The text expresses negative feelings.</p>
            </div>
            """, unsafe_allow_html=True)

        else:
            st.markdown(f"""
            <div class="result-card" style="background:linear-gradient(135deg,#d7e1ec,#f2f4f7);">
                <h1>😐 Neutral</h1>
                <p><b>Confidence:</b> {conf:.1f}%</p>
                <p>The message looks neutral and balanced.</p>
            </div>
            """, unsafe_allow_html=True)
