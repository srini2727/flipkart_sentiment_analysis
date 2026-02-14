# Flipkart Reviews Sentiment Analysis 🛍️

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-App-ff4b4b)
[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/)

An end-to-end Sentiment Analysis Machine Learning project that classifies Flipkart product reviews as Positive or Negative. This project demonstrates a full NLP pipeline—from data cleaning and preprocessing to model training and interactive deployment using Streamlit.

## 🚀 Key Features

*   **Interactive Dashboard**: A user-friendly web interface built with [Streamlit](https://streamlit.io/).
*   **Dual Analysis Approach**:
    *   **Lexicon-Based**: Uses **VADER** (Valence Aware Dictionary and sEntiment Reasoner) for rule-based sentiment scoring.
    *   **Machine Learning**: Implements a **Logistic Regression** model trained on TF-IDF features for robust classification.
*   **Advanced Preprocessing**: Includes text cleaning, stopword removal, and **Lemmatization** for high-quality input data.
*   **Data Visualization**: features interactive charts (Plotly) and WordClouds to explore data distribution and common terms.
*   **Real-time Prediction**: Users can type their own reviews and get instant sentiment predictions from both VADER and the ML model.

## 📸 Screenshots

### 1. Data Overview & Word Cloud
Explore the raw data and see the most frequent words used in reviews.
![Data Overview](screenshots/Data%20Overview.png)

### 2. Rating Distribution
Visualizing how customers rate products on Flipkart.
![Rating Distribution](screenshots/Rating%20Distribution.png)

### 3. Model Analysis & Visualizations
Comprehensive view of the Logistic Regression model performance and global sentiment predictions.
![Model Viz](screenshots/Model%20Visualization.png)
![Predicted Sentiment](screenshots/Predicted%20Sentiment.png)

### 4. Live Prediction Demo
Test the model with your own custom text input!
![Prediction Demo](screenshots/Navigation%20&%20Test.png)


## 🛠️ Tech Stack

*   **Language**: Python
*   **Web Framework**: Streamlit
*   **Machine Learning**: Scikit-Learn (Logistic Regression, TF-IDF Vectorizer)
*   **NLP**: NLTK (VADER, WordNet Lemmatizer, Stopwords)
*   **Data Processing**: Pandas, NumPy
*   **Visualization**: Plotly Express, Matplotlib, WordCloud

## 📂 Project Structure

```bash
flipkart_sentiment/
├── app.py                # Main Streamlit application entry point
├── src/
│   ├── preprocessing.py  # Text cleaning and Lemmatization logic
│   ├── analysis.py       # VADER scoring and Model Training pipeline
│   └── __init__.py       # Package initialization
├── data/                 # Dataset storage (excluded from git)
├── requirements.txt      # Project dependencies
└── README.md             # Project documentation
```

## ⚡ Installation & Usage

1.  **Clone the Repository**
    ```bash
    git clone https://github.com/yourusername/flipkart-sentiment-analysis.git
    cd flipkart-sentiment-analysis
    ```

2.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the Application**
    ```bash
    streamlit run app.py
    ```

4.  **Access the App**
    Open your browser and navigate to `http://localhost:8501`.

## 📊 Methodology

1.  **Data Cleaning**: Raw text is lowercased; URLs, punctuation, and numbers are removed.
2.  **Preprocessing**: Stopwords are filtered out, and words are lemmatized to their root form.
3.  **Labeling**: 
    *   Reviews with 4-5 stars are labeled **Positive (1)**.
    *   Reviews with 1-2 stars are labeled **Negative (0)**.
    *   Neutral reviews (3 stars) are excluded from binary classification training.
4.  **Model Training**: A pipeline converts text to TF-IDF vectors and trains a Logistic Regression classifier.

## 📈 Results

The custom Logistic Regression model achieves robust accuracy on the test set, effectively distinguishing between positive and negative feedback even in complex sentences.


---

