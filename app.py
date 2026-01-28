import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer

# Initialize stemmer
ps = PorterStemmer()

# Download necessary NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

# Text preprocessing function
def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

# Load vectorizer and model
tfid = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

# ------------------- UI Section -------------------

# Page config
st.set_page_config(page_title="SMS Spam Classifier", page_icon="📩", layout="centered")

# Custom CSS styling
st.markdown("""
    <style>
    .main {
        background-color: #f9f9f9;
    }
    h1 {
        color: #2E86C1;
        text-align: center;
        font-family: 'Arial Black', sans-serif;
    }
    .stTextArea textarea {
        border: 2px solid #2E86C1;
        border-radius: 10px;
    }
    .stButton>button {
        background-color: #2E86C1;
        color: white;
        border-radius: 10px;
        font-size: 16px;
        padding: 8px 20px;
    }
    .stButton>button:hover {
        background-color: #1B4F72;
        color: #f1f1f1;
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.title("📩 SMS Spam Classifier")

# Description
st.markdown("### 🔍 Enter an SMS message below and let the classifier decide whether it's **Spam** or **Not Spam**.")

# Input box
input_sms = st.text_area("✉️ Type your SMS message here:", height=150)

# Prediction button
if st.button('🚀 Predict'):
    if input_sms.strip() != "":
        # 1. Preprocess
        transform_sms = transform_text(input_sms)
        # 2. Vectorize
        vector_input = tfid.transform([transform_sms])
        # 3. Predict
        result = model.predict(vector_input)[0]
        # 4. Display result with style
        if result == 1:
            st.error("⚠️ This message looks like **Spam**!")
        else:
            st.success("✅ This message looks **Not Spam**.")
    else:
        st.warning("⚠️ Please enter a message before predicting.")
