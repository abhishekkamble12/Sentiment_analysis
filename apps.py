# import streamlit as st
# import pandas as pd
# import joblib
# import nltk

# # 1. Setup NLTK (VADER is still needed for feature engineering inside the pipeline)
# @st.cache_resource
# def setup_nltk():
#     try:
#         nltk.data.find('sentiment/vader_lexicon.zip')
#     except LookupError:
#         nltk.download('vader_lexicon')
#     from nltk.sentiment.vader import SentimentIntensityAnalyzer
#     return SentimentIntensityAnalyzer()

# sid = setup_nltk()

# # 2. Load Model
# model = joblib.load('model.pkl')
# le = joblib.load('label_encoder.pkl')

# # 3. Helper to get VADER score (Pipeline needs this column)
# def get_vader_score(text):
#     return sid.polarity_scores(text)['compound']

# # 4. Streamlit UI
# st.title("🤖 Simple Sentiment Analyzer")

# with st.form("my_form"):
#     text_input = st.text_area("Enter comment:")
#     platform = st.selectbox("Platform", ["Twitter", "Facebook", "Instagram", "Website"])
#     time = st.selectbox("Time", ["morning", "noon", "night"])
#     submitted = st.form_submit_button("Predict")

# if submitted and text_input:
#     # Prepare data exactly as the training pipeline expects
#     # Note: We do NOT need to manually clean text or TF-IDF here. The pipeline does it.
#     df = pd.DataFrame({
#         'cleaned_text': [text_input],  # Pipeline will handle cleaning if built correctly, or pass raw text
#         'Platform': [platform],
#         'Time of Tweet': [time],
#         'vader_score': [get_vader_score(text_input)]
#     })
    
#     # Predict
#     pred = model.predict(df)[0]
#     label = le.inverse_transform([pred])[0]
    
#     st.success(f"Sentiment: **{label}**")