Text Sentiment Analysis Project

This project is a Text Sentiment Analysis system that classifies movie reviews into Positive, Negative, or Neutral sentiments using Machine Learning and Deep Learning techniques.
The project uses Logistic Regression as a baseline model and an LSTM-based Deep Learning model, along with rule-based NLP logic, to improve prediction accuracy.

The final model is deployed using Streamlit for real-time sentiment analysis.

📌 Project Overview

Sentiment analysis is an important task in Natural Language Processing (NLP) that identifies emotions or opinions expressed in text.

In this project:

i) Logistic Regression is used as a classical machine learning baseline

ii) LSTM (Long Short-Term Memory) is used for deep learning-based sentiment prediction

iii) Rule-based techniques are applied to handle negations, strong sentiment words, and long reviews

This hybrid approach helps reduce incorrect predictions and improves overall performance.

🎯 Objectives

i) To perform sentiment analysis on movie reviews

ii) To compare Machine Learning and Deep Learning models

iii) To improve accuracy using rule-based corrections

iv) To deploy the model using a web interface

🛠 Technologies Used

i) Python

ii) Logistic Regression

iii) TensorFlow / Keras

iv) LSTM Neural Network

v) Natural Language Processing (NLP)

vi) Streamlit

vii) IMDb Movie Reviews Dataset

viii) Pickle

📂 Project File Structure

├── app.py                                           # Streamlit application

├── tokenizer.pkl                                    # Saved tokenizer

├── sentiment_model.keras                            # Trained LSTM model

├── IMDB_Sentiment_analysis.ipynb                    # Logistic Regression & LSTM training

├── requirements.txt                                 # Dependencies

└── README.md                                        # Project documentation


⚙️ Models Used

1️⃣ Logistic Regression

Used as a baseline Machine Learning model

Uses TF-IDF vectorization

Fast and interpretable

Helps compare traditional ML vs Deep Learning performance

2️⃣ LSTM (Deep Learning Model)

Learns sequential patterns in text

Better performance on long reviews

Handles complex sentence structures

3️⃣ Rule-Based NLP Layer

Handles negations like "not good", "not bad"

Detects strong sentiment dominance

Reduces false predictions in long reviews

🚀 How the System Works

User enters a movie review

Text is cleaned and preprocessed

Rule-based sentiment checks are applied

If rules do not match, the LSTM model predicts sentiment

Final sentiment and confidence score are displayed

📊 Model Evaluation

Evaluated using confusion matrix

Logistic Regression used for comparison

Hybrid approach improved robustness

<img width="635" height="760" alt="image" src="https://github.com/user-attachments/assets/4896ce06-db81-4312-a47c-9bfd5dff1daa" />



✅ Key Features

Combines ML + DL models

Handles negation and mixed sentiment

Supports long reviews

User-friendly Streamlit interface

Academic and real-world applicable

🔮 Future Enhancements

Add Transformer-based models (BERT)

Multi-class emotion detection

Cloud deployment

Performance comparison dashboard

👩‍💻 Author

Sriparna Majumder

