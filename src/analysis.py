import pandas as pd
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report

# Ensure VADER lexicon is downloaded
try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError:
    nltk.download('vader_lexicon')

def get_vader_scores(text):
    """
    Returns the VADER polarity scores for a given text.
    """
    sid = SentimentIntensityAnalyzer()
    return sid.polarity_scores(text)

def train_sentiment_model(df):
    """
    Trains a Logistic Regression model on the dataset.
    Assumes df has 'Clean_Review' and 'Rating' columns.
    
    Target:
    1 if Rating >= 4 (Positive)
    0 if Rating <= 2 (Negative)
    Neutral ratings (3) are excluded for binary classification training.
    """
    # Filter for positive and negative only for better binary classification
    df_model = df[df['Rating'] != 3].copy()
    
    # Label: 1 for Positive (4,5), 0 for Negative (1,2)
    df_model['target'] = df_model['Rating'].apply(lambda x: 1 if x > 3 else 0)
    
    X = df_model['Clean_Review'].astype(str)
    y = df_model['target']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Create Pipeline
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=5000)),
        ('clf', LogisticRegression())
    ])
    
    pipeline.fit(X_train, y_train)
    
    y_pred = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    return pipeline, accuracy, classification_report(y_test, y_pred)
