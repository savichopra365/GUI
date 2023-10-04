from nltk.sentiment.vader import SentimentIntensityAnalyzer
import streamlit as st
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

import nltk


# Download the VADER lexicon
nltk.download('vader_lexicon')

# Now you can use VADER


sia = SentimentIntensityAnalyzer()

# Streamlit app
st.title("Sentiment Analysis App (user input)")

# User input
user_input = st.text_input("Enter a sentence:")

if user_input:
    # Perform sentiment analysis
    sentiment_scores = sia.polarity_scores(user_input)

    # Interpret the sentiment scores
    if sentiment_scores['compound'] > 0.35:
        sentiment = "Positive"
    elif sentiment_scores['compound'] < -0.25:
        sentiment = "Negative"
    else:
        sentiment = 'Nuetral'

    st.write("Sentiment:", sentiment)
    st.write("Sentiment Scores:", sentiment_scores)
