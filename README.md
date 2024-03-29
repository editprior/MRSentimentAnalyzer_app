# Movie Review Sentiment Analysis 🍿

## Overview
This Streamlit web application analyzes the sentiment of movie reviews using Natural Language Processing (NLP) techniques and a machine learning model. Users can input a movie review, and the app will classify the sentiment as positive or negative, providing insights into the overall tone of the review.
Check out [live demo](https://mrsentimentanalyzerapp-6tip8rygftjbnpqmfyjufc.streamlit.app/). 

## Features
- Input a movie review and receive sentiment analysis results.
- Visualize the analyzed review with highlighted positive and negative words.
- Display an explanation of the sentiment analysis using LIME (Local Interpretable Model-agnostic Explanations).

## Dependencies
This application utilizes several Python libraries to perform sentiment analysis on movie reviews:
- Streamlit for building the user interface.
- Pandas for data manipulation.
- scikit-learn for machine learning tasks such as splitting data, building pipelines, and evaluating models.
- NLTK for natural language processing tasks such as removing stopwords.
- Joblib for saving and loading machine learning models.
- NumPy for numerical operations.
- Matplotlib for generating visualizations.
- Other libraries such as String, RandomForestClassifier, TfidfVectorizer, LimeTextExplainer, and re are also used for various text processing and visualization tasks.

## Installation
If any of these libraries are not already installed, they can be installed using pip:
```pip install streamlit pandas scikit-learn joblib nltk matplotlib```

Additionally, NLTK's stopwords data needs to be downloaded:
```import nltk```
```nltk.download('stopwords')```
