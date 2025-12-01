import streamlit as st
import pickle
import re
import os
import time

# -----------------------------------------------------------------------------
# 1. PAGE CONFIGURATION (Must be the first command)
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Sentiment AI",
    page_icon="✨",
    layout="centered",
    initial_sidebar_state="expanded"
)

# -----------------------------------------------------------------------------
# 2. CUSTOM CSS FOR AESTHETICS
# -----------------------------------------------------------------------------
st.markdown("""
    <style>
    /* Main Background & Font */
    .stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        font-family: 'Helvetica Neue', sans-serif;
    }
    
    /* Title Styling */
    .title-text {
        color: #2c3e50;
        text-align: center;
        font-size: 50px;
        font-weight: 800;
        margin-bottom: 10px;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    
    /* Subtitle Styling */
    .subtitle-text {
        color: #5d6d7e;
        text-align: center;
        font-size: 18px;
        margin-bottom: 40px;
    }

    /* Text Area Styling */
    .stTextArea textarea {
        background-color: #ffffff;
        border-radius: 12px;
        border: 2px solid #dfe6e9;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        font-size: 16px;
        padding: 15px;
        transition: border-color 0.3s ease;
    }
    .stTextArea textarea:focus {
        border-color: #a29bfe;
        box-shadow: 0 0 0 2px rgba(162, 155, 254, 0.2);
    }

    /* Button Styling */
    .stButton>button {
        width: 100%;
        background: linear-gradient(90deg, #6c5ce7, #a29bfe);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 12px 24px;
        font-size: 18px;
        font-weight: 600;
        cursor: pointer;
        transition: transform 0.2s, box-shadow 0.2s;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(108, 92, 231, 0.3);
        color: white;
    }

    /* Result Card Styling */
    .result-box {
        padding: 25px;
        border-radius: 15px;
        text-align: center;
        margin-top: 20px;
        box-shadow: 0 10px 20px rgba(0,0,0,0.1);
        animation: fadeIn 0.5s ease-in;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    </style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 3. HELPER FUNCTIONS
# -----------------------------------------------------------------------------

# Cleaning Function (Exact same as training)
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    return text

# Load Model Function with Caching
@st.cache_resource
def load_model():
    if not os.path.exists('sentiment_model.pkl'):
        return None
    with open('sentiment_model.pkl', 'rb') as f:
        model = pickle.load(f)
    return model

# -----------------------------------------------------------------------------
# 4. MAIN APP UI
# -----------------------------------------------------------------------------

# Sidebar
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2065/2065224.png", width=100)
    st.title("About App")
    st.info(
        """
        This AI-powered app analyzes the sentiment of your text.
        
        **How it works:**
        1. Enter your text.
        2. Click Analyze.
        3. See if it's Positive, Negative, or Neutral.
        """
    )
    st.write("---")
    st.caption("Made with ❤️ using Streamlit")

# Main Content
st.markdown('<div class="title-text">Sentiment AI ✨</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle-text">Uncover the emotions hidden in your text instantly.</div>', unsafe_allow_html=True)

# Load Model
model = load_model()

# Check if model exists
if model is None:
    st.error("🚨 **Model File Missing!** Please download `sentiment_model.pkl` from Colab and place it in this folder.")
else:
    # Input Area
    user_input = st.text_area("✍️ Enter your text here:", height=150, placeholder="Type something amazing...")

    # Layout for Button
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        analyze_button = st.button("🚀 Analyze Sentiment")

    # Analysis Logic
    if analyze_button:
        if user_input.strip():
            with st.spinner('Thinking... 🤔'):
                # Simulate processing time for effect
                time.sleep(0.8) 
                
                cleaned_input = clean_text(user_input)
                prediction = model.predict([cleaned_input])[0]
                
                # Confidence Score (Optional, if model supports it)
                try:
                    probs = model.predict_proba([cleaned_input])[0]
                    confidence = max(probs) * 100
                except:
                    confidence = 0

            # Display Result with Custom HTML/CSS
            if prediction == 'positive':
                st.markdown(
                    f"""
                    <div class="result-box" style="background: linear-gradient(135deg, #a8ff78 0%, #78ffd6 100%); color: #1e5631;">
                        <h1 style="margin:0;">😊 Positive</h1>
                        <p style="margin:0;">Confidence: {confidence:.1f}%</p>
                        <p style="font-size: 14px; margin-top: 5px;">This text spreads good vibes!</p>
                    </div>
                    """, 
                    unsafe_allow_html=True
                )
                st.balloons()
                
            elif prediction == 'negative':
                st.markdown(
                    f"""
                    <div class="result-box" style="background: linear-gradient(135deg, #ff9966 0%, #ff5e62 100%); color: #561e1e;">
                        <h1 style="margin:0;">😠 Negative</h1>
                        <p style="margin:0;">Confidence: {confidence:.1f}%</p>
                        <p style="font-size: 14px; margin-top: 5px;">This text seems a bit harsh.</p>
                    </div>
                    """, 
                    unsafe_allow_html=True
                )
                
            else:
                st.markdown(
                    f"""
                    <div class="result-box" style="background: linear-gradient(135deg, #e0eafc 0%, #cfdef3 100%); color: #3b4d61;">
                        <h1 style="margin:0;">😐 Neutral</h1>
                        <p style="margin:0;">Confidence: {confidence:.1f}%</p>
                        <p style="font-size: 14px; margin-top: 5px;">This text is balanced and objective.</p>
                    </div>
                    """, 
                    unsafe_allow_html=True
                )
        else:
            st.warning("⚠️ Please type something first!")