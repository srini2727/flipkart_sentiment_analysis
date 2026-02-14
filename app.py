import streamlit as st
import pandas as pd
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from src.preprocessing import clean_text
from src.analysis import get_vader_scores, train_sentiment_model

# Page Config
st.set_page_config(page_title="Flipkart Sentiment Analysis", layout="wide")

st.title("🛍️ Flipkart Reviews Sentiment Analysis")
st.markdown("Analyze customer sentiment from Flipkart reviews using VADER and Machine Learning.")

# Load Data
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("data/flipkart_reviews.csv")
        return df
    except FileNotFoundError:
        st.error("Data file not found. Please ensure 'data/flipkart_reviews.csv' exists.")
        return None

df = load_data()

if df is not None:
    # Sidebar
    st.sidebar.header("Navigation")
    option = st.sidebar.radio("Go to:", ["Data Exploration", "Sentiment Analysis", "Prediction Demo"])

    # Preprocessing (Cached)
    @st.cache_data
    def get_clean_data(data):
        with st.spinner("Preprocessing data (Cleaning & Lemmatizing)... this may take a moment"):
            data['Clean_Review'] = data['Review'].apply(clean_text)
        return data

    if "Clean_Review" not in df.columns:
        df = get_clean_data(df)

    # --- Data Exploration ---
    if option == "Data Exploration":
        st.header("📊 Data Overview")
        st.write(f"Total Reviews: {len(df)}")
        st.dataframe(df.head())

        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Rating Distribution")
            fig = px.pie(df, names='Rating', title='Distribution of Ratings', hole=0.4)
            st.plotly_chart(fig)

        with col2:
            st.subheader("Word Cloud")
            all_text = " ".join(review for review in df.Clean_Review)
            wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_text)
            fig_wc, ax = plt.subplots()
            ax.imshow(wordcloud, interpolation='bilinear')
            ax.axis("off")
            st.pyplot(fig_wc)

    # --- Sentiment Analysis ---
    elif option == "Sentiment Analysis":
        st.header("📈 Sentiment Analysis")
        
        st.subheader("VADER Sentiment Scores")
        if st.checkbox("Show VADER Scores for sample"):
            sample = df.sample(5)
            sample['VADER_Scores'] = sample['Review'].apply(get_vader_scores)
            st.write(sample[['Review', 'Rating', 'VADER_Scores']])

        st.divider()

        st.subheader("Machine Learning Model (Logistic Regression)")
        if 'model' not in st.session_state:
            with st.spinner("Training Model..."):
                pipeline, accuracy, report = train_sentiment_model(df)
                st.session_state['model'] = pipeline
                st.session_state['accuracy'] = accuracy
                st.session_state['report'] = report
        
        col_acc, col_rep = st.columns([1, 2])
        with col_acc:
            st.metric("Model Accuracy", f"{st.session_state['accuracy']*100:.2f}%")
        with col_rep:
             with st.expander("View Classification Report"):
                st.code(st.session_state['report'])

        st.divider()
        st.subheader("Model Visualizations")
        
        # Chart 1: VADER Compound Score Distribution
        st.markdown("#### 1. VADER Sentiment Distribution")
        st.caption("Distribution of VADER compound scores. Scores > 0.05 are Positive, < -0.05 are Negative.")
        
        # We need to compute stats for the whole DF for the chart if not already
        if 'vader_compound' not in df.columns:
            with st.spinner("Calculating VADER scores for visualization..."):
                df['vader_compound'] = df['Clean_Review'].apply(lambda x: get_vader_scores(x)['compound'])
        
        fig_hist = px.histogram(df, x="vader_compound", nbins=50, title="Distribution of VADER Compound Scores",
                                color_discrete_sequence=['#636EFA'])
        st.plotly_chart(fig_hist, use_container_width=True)

        # Chart 2: Predicted Sentiment Counts (ML Model)
        st.markdown("#### 2. Predicted Sentiment Counts (ML Model)")
        st.caption("How the Logistic Regression model classifies the entire dataset.")
        
        if 'predicted_sentiment' not in df.columns:
             with st.spinner("Classifying all reviews..."):
                 if 'model' not in st.session_state:
                     # This should be trained by now, but safety check
                     pipeline, _, _ = train_sentiment_model(df)
                     st.session_state['model'] = pipeline
                 
                 predictions = st.session_state['model'].predict(df['Clean_Review'].astype(str))
                 df['predicted_sentiment'] = ["Positive" if p == 1 else "Negative" for p in predictions]
        
        pred_counts = df['predicted_sentiment'].value_counts().reset_index()
        pred_counts.columns = ['Sentiment', 'Count']
        
        fig_bar = px.bar(pred_counts, x='Sentiment', y='Count', color='Sentiment', 
                         title="Total Reviews by Predicted Sentiment",
                         color_discrete_map={"Positive": "#00CC96", "Negative": "#EF553B"})
        st.plotly_chart(fig_bar, use_container_width=True)


    # --- Prediction Demo ---
    elif option == "Prediction Demo":
        st.header("🧠 Test the Model")
        user_input = st.text_area("Enter a review to analyze:", "This product is amazing! good quality.")
        
        if st.button("Analyze Sentiment"):
            if user_input:
                # 1. VADER
                vader_scores = get_vader_scores(user_input)
                
                # 2. ML Prediction
                clean_input = clean_text(user_input)
                if 'model' not in st.session_state:
                     pipeline, _, _ = train_sentiment_model(df)
                     st.session_state['model'] = pipeline
                
                prediction = st.session_state['model'].predict([clean_input])[0]
                proba = st.session_state['model'].predict_proba([clean_input])[0]
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.info("VADER Analysis")
                    st.write(vader_scores)
                    if vader_scores['compound'] >= 0.05:
                        st.success("Overall: Positive 😊")
                    elif vader_scores['compound'] <= -0.05:
                        st.error("Overall: Negative 😠")
                    else:
                        st.warning("Overall: Neutral 😐")

                with col2:
                    st.info("ML Model Prediction")
                    if prediction == 1:
                        st.success(f"Prediction: Positive ({proba[1]*100:.2f}%)")
                    else:
                        st.error(f"Prediction: Negative ({proba[0]*100:.2f}%)")
