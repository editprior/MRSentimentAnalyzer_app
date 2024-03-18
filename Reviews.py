import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline # Importing Pipeline for creating a data processing pipeline
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from joblib import dump, load
import string
import nltk # Importing NLTK library for natural language processing tasks
from nltk.corpus import stopwords
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from lime.lime_text import LimeTextExplainer
import re # Importing re module for regular expressions
from wordcloud import WordCloud # Importing WordCloud for creating word clouds
import matplotlib.pyplot as plt
import zipfile

# List of CSV files to concatenate
csv_files = ['IMDB Dataset.csv', '2.csv', '3.csv', '4.csv', '5.csv']
# Read each CSV file into a DataFrame and store them in a list
dfs = [pd.read_csv(file) for file in csv_files]

# Concatenate the DataFrames along the rows (axis=0)
reviews_df = pd.concat(dfs, ignore_index=True)

# Disable PyplotGlobalUseWarning
st.set_option('deprecation.showPyplotGlobalUse', False)

nltk.download('stopwords')

# Load model
@st.cache_data
def load_model():
    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(reviews_df['review'], reviews_df['sentiment'], stratify=reviews_df['sentiment'], train_size=0.8, random_state=123)
    # Create a pipeline for text classification
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=10000, stop_words='english')),
        ('classifier', RandomForestClassifier(n_estimators=50, max_depth=20))
    ]) 

    # Fit the pipeline on the training data
    pipeline.fit(X_train, y_train)

    # Make predictions on training and testing sets
    y_train_preds = pipeline.predict(X_train)
    y_test_preds = pipeline.predict(X_test)

    # Save the pipeline model
    dump(pipeline, 'movie_review_classifier.joblib')

    return pipeline, X_test

pipeline, X_test = load_model()

# Function to preprocess text
def preprocess_text(text):
    text = text.lower() # Convert text to lowercase
    text = text.translate(str.maketrans('', '', string.punctuation)) # Remove punctuation
    text = ' '.join([word for word in text.split() if word not in stopwords.words('english')]) # Remove stopwords
    return text

# Function to predict sentiment
def predict_sentiment(text):
    preprocessed_text = preprocess_text(text) # Preprocess the text

    # Predict sentiment using the trained model
    prediction_probs = pipeline.predict_proba([preprocessed_text])[0]
    predicted_class = pipeline.predict([preprocessed_text])[0]
    confidence = np.max(prediction_probs) * 100  # Convert to percentage
    return predicted_class, confidence

# Function to create colored text with highlighted words
def create_colored_review(review, word_contribution):
    for word, contribution in word_contribution:
        if contribution > 0:
            # Highlight positive words in green
            review = re.sub(r'\b' + re.escape(word) + r'\b', f'<span style="background-color: #d4edda">{word}</span>', review)
        else:
            # Highlight negative words in red
            review = re.sub(r'\b' + re.escape(word) + r'\b', f'<span style="background-color: #f8d7da">{word}</span>', review)
    return review

# Function to create word cloud
def create_word_cloud(word_contribution):
    wordcloud_dict = {word: abs(contribution) for word, contribution in word_contribution} # Create a dictionary of words and their contributions
    wordcloud = WordCloud(width=1000, height=800, background_color ='white').generate_from_frequencies(wordcloud_dict) # Generate a word cloud from the dictionary
    # Plot the word cloud
    plt.figure(figsize=(15, 12))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title('Word Cloud of Significant Words', fontsize=20)
    st.pyplot()

# Streamlit app
st.title("Movie Review Sentiment Analysis🍿")
# Description of the app and its usefulness
# Description of the app and its usefulness along with References
st.markdown("""
    <div style='background-color: #f9f9f9; padding: 20px; border-radius: 10px;'>
        <h2 style='color: #333333; font-size: 24px;'>🎬 Welcome to Movie Review Sentiment Analyzer!</h2>
        <p style='font-size: 18px; color: #555555;'>
            Have you ever wondered what others think about a movie before watching it? Are you a filmmaker looking to understand audience reactions to your latest project? Look no further! Our app is here to help.
        </p>
        <h2 style='color: #333333; font-size: 24px;'>🔍 How It Works</h2>
        <p style='font-size: 18px; color: #555555;'>
            Simply enter the review of the movie you're interested in, and our powerful machine learning model will analyze it to determine whether the sentiment expressed is positive or negative. The app visualizes the results, providing valuable insights at a glance.
        </p>
        <h2 style='color: #333333; font-size: 24px;'>🎥 Use Case: Choosing the Perfect Movie</h2>
        <p style='font-size: 18px; color: #555555;'>
            Planning a movie night with friends can be tricky when you're unsure which film to choose. Our app simplifies the process by allowing you to analyze multiple movie reviews quickly. With just a few clicks, you can select the movie with the most positive sentiment, ensuring everyone enjoys the viewing experience. No need to read entire reviews and risk spoilers — our Analyzer does the sentiment analysis for you. Say goodbye to wasting time on lengthy reviews and hello to hassle-free movie nights with Movie Review Sentiment Analyzer!
        </p>
        <h2 style='color: #333333; font-size: 24px;'>References</h2>
        <ol>
            <li><a href="https://www.youtube.com/watch?v=Pp2zYby0gtc&ab_channel=CoderzColumn">Very helpful tutorial! Huge shoutout to the creator of this tutorial. Reviews Classification Dashboard using STREAMLIT</a></li>
            <li><a href="https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews">IMDB Dataset of 50K Movie Reviews</a></li>
            <li><a href="https://coderzcolumn.com/tutorials/machine-learning/scikit-learn-sklearn-ensemble-learning-bagging-and-random-forests">Random Forest Guide</a></li>
            <li><a href="https://coderzcolumn.com/tutorials/machine-learning/feature-extraction-from-text-data-using-scikit-learn-sklearn">TfIDf Guide</a></li>
            <li><a href="https://coderzcolumn.com/tutorials/machine-learning/model-evaluation-scoring-metrics-scikit-learn-sklearn">ML Metrics Guide</a></li>
            <li><a href="https://coderzcolumn.com/tutorials/machine-learning/how-to-use-lime-to-understand-sklearn-models-predictions">LIME Guide</a></li>
            <li><a href="https://coderzcolumn.com/tutorials/python/joblib-parallel-processing-in-python">Joblib</a></li>
            <li><a href="https://www.youtube.com/watch?v=a6JLETUoA-g&ab_channel=Pythonology">How to make a WordCloud using Python | Streamlit</a></li>
            <li><a href="https://medium.com/@nimritakoul01/nlp-with-python-part-2-nltk-cc6fe52f1a1a">NLP with Python Part 2 NLTK</a></li>
            <li><a href="https://www.pythonprog.com/natural-language-processing-nlp-in-python-with-nltk/">Natural Language Processing (NLP) in Python with NLTK</a></li>
            <li><a href="https://chat.openai.com/">Chat GPT</a></li>
        </ol>
    </div>
""", unsafe_allow_html=True)

# Text input for user
review = st.text_area(label="Enter Review Here:", height=20)

submit = st.button("Classify")

if submit and review:
    col1, col2, col3 = st.columns(3, gap="medium")
    word_count = len(review.split())
    st.text(f"Word Count: {word_count}")
    predicted_class, confidence = predict_sentiment(review)

    with col1:
        if predicted_class == 'positive':
            st.markdown("### Prediction: <span style='color:green'>{}</span>".format(predicted_class), unsafe_allow_html=True)
        else:
            st.markdown("### Prediction: <span style='color:red'>{}</span>".format(predicted_class), unsafe_allow_html=True)
        st.metric(label="Confidence", value="{:.2f}%".format(confidence))
    
    # Explanation code
    # Creating a LimeTextExplainer object to generate explanations for the model predictions
    explainer = LimeTextExplainer(class_names=pipeline.named_steps['classifier'].classes_)

    # Function to predict the probabilities for the input text
    def predict_fn(text_list):
        return pipeline.predict_proba(text_list)

    # Generating an explanation for the instance (review) using the explainer object
    explanation = explainer.explain_instance(review, predict_fn, num_features=50)
    word_contribution = explanation.as_list() # Extracting word contributions from the explanation
    
    # Display analyzed output with highlighted text
    modified_review = create_colored_review(review, word_contribution)
    st.markdown(modified_review, unsafe_allow_html=True)

    # Displaying the explanation visualization
    with col2:
        fig = explanation.as_pyplot_figure()
        fig.set_figheight(9)
        st.pyplot(fig, use_container_width=True)

    # Display word cloud
    with col3:
        create_word_cloud(word_contribution)
