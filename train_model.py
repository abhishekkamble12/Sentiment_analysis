import pandas as pd
import re
import pickle
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

# 1. Clean up old file if it exists
if os.path.exists("sentiment_model.pkl"):
    os.remove("sentiment_model.pkl")
    print("Old corrupted file deleted.")

# 2. Load Data
print("Loading data...")
try:
    df = pd.read_csv('sentiment_analysis.csv')
except FileNotFoundError:
    print("Error: 'sentiment_analysis.csv' not found. Make sure it is in this folder.")
    exit()

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    return text

df['clean_text'] = df['text'].apply(clean_text)

# 3. Train
print("Training model...")
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(ngram_range=(1, 2), stop_words='english')),
    ('clf', LogisticRegression(C=10, max_iter=1000, random_state=42))
])
pipeline.fit(df['clean_text'], df['sentiment'])

# 4. Save
print("Saving model to 'sentiment_model.pkl'...")
with open('sentiment_model.pkl', 'wb') as f:
    pickle.dump(pipeline, f)

# 5. Verify (The Self-Test)
print("Verifying the new file...")
try:
    with open('sentiment_model.pkl', 'rb') as f:
        test_model = pickle.load(f)
    print("✅ SUCCESS! The model file is valid and ready.")
except Exception as e:
    print(f"❌ FAILED: Something went wrong. Error: {e}")