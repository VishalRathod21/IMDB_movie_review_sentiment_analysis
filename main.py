# Import libraries - standard imports first
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# Then Streamlit import (must come before any Streamlit commands)
import streamlit as st

# Then other imports
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model

# Set page config - MUST BE FIRST STREAMLIT COMMAND
st.set_page_config(
    page_title="IMDB Sentiment Analysis",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load the IMDB dataset word index
@st.cache_data
def load_word_index():
    word_index = imdb.get_word_index()
    reverse_word_index = {value: key for key, value in word_index.items()}
    return word_index, reverse_word_index

word_index, reverse_word_index = load_word_index()

# Load the pre-trained model
@st.cache_resource
def load_model_from_file():
    return load_model('simple_rnn_imdb.h5')

model = load_model_from_file()

# Helper Functions
def decode_review(encoded_review):
    return ' '.join([reverse_word_index.get(i - 3, '?') for i in encoded_review])

def preprocess_text(text):
    words = text.lower().split()
    encoded_review = [word_index.get(word, 2) + 3 for word in words]
    padded_review = sequence.pad_sequences([encoded_review], maxlen=500)
    return padded_review

def generate_wordcloud(text, sentiment):
    wordcloud = WordCloud(
        width=800,
        height=400,
        background_color='white',
        colormap='cool' if sentiment == 'Positive' else 'autumn',
        stopwords={'the', 'and', 'this', 'that', 'was', 'for', 'with'}
    ).generate(text)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    return fig

# Main App
st.title('🎬 IMDB Movie Review Sentiment Analysis')
st.markdown("""
    <style>
    .main {padding-top: 1rem;}
    .stTextArea textarea {font-size: 16px;}
    .stButton button {width: 100%;}
    </style>
    """, unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("About")
    st.write("""
    This app uses a pre-trained RNN model to analyze movie review sentiment.
    The model was trained on the IMDB movie review dataset.
    """)
    
    st.divider()
    st.subheader("Example Reviews")
    examples = {
        "Positive": "This movie was fantastic! The acting was superb and the story kept me engaged throughout.",
        "Negative": "I was disappointed. The plot was predictable and the acting felt forced.",
        "Mixed": "The movie had great moments but overall was just okay. Some scenes dragged on too long."
    }
    
    for sentiment, text in examples.items():
        if st.button(f"{sentiment} Example"):
            st.session_state.user_input = text

# Main Content
col1, col2 = st.columns([2, 1])

with col1:
    user_input = st.text_area(
        "Enter your movie review:",
        height=200,
        value=st.session_state.get("user_input", ""),
        placeholder="Type or paste a movie review here..."
    )
    
    if st.button("Analyze Sentiment", type="primary"):
        if not user_input.strip():
            st.warning("Please enter a review to analyze")
        else:
            with st.spinner("Processing your review..."):
                try:
                    # Preprocess and predict
                    processed_input = preprocess_text(user_input)
                    prediction = model.predict(processed_input)
                    confidence = float(prediction[0][0])
                    sentiment = "Positive" if confidence > 0.5 else "Negative"
                    confidence_pct = round((confidence if sentiment == "Positive" else 1-confidence) * 100, 1)
                    
                    # Display results
                    st.subheader("Analysis Results")
                    
                    # Sentiment indicator
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Sentiment", sentiment)
                    with col2:
                        st.metric("Confidence", f"{confidence_pct}%")
                    
                    # Visualizations
                    tab1, tab2 = st.tabs(["Confidence Meter", "Word Cloud"])
                    
                    with tab1:
                        fig, ax = plt.subplots(figsize=(8, 2))
                        ax.barh([0], [confidence], color='#4CAF50' if sentiment == "Positive" else '#F44336', height=0.5)
                        ax.set_xlim(0, 1)
                        ax.set_xticks([0, 0.5, 1])
                        ax.set_xticklabels(["0% (Negative)", "50%", "100% (Positive)"])
                        ax.set_yticks([])
                        ax.set_title("Sentiment Confidence")
                        st.pyplot(fig)
                    
                    with tab2:
                        st.pyplot(generate_wordcloud(user_input, sentiment))
                    
                    # Debug info
                    with st.expander("Review Details"):
                        st.write(f"**Processed words:** {len(processed_input[0])} tokens")
                        st.write(f"**Raw prediction value:** {confidence:.4f}")
                        
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")

with col2:
    st.subheader("How It Works")
    st.markdown("""
    1. Type or paste a movie review
    2. Click "Analyze Sentiment"
    3. View the results including:
       - Sentiment (Positive/Negative)
       - Confidence level
       - Word frequency visualization
    """)
    
    st.divider()
    st.subheader("Tips for Better Results")
    st.markdown("""
    - Write at least 2-3 sentences
    - Include emotional words (great, terrible, etc.)
    - Avoid very short or vague reviews
    """)

# Footer
st.divider()
st.caption("""
    Note: This is a machine learning model and may not always be accurate. 
    Results are for educational purposes only.
""")