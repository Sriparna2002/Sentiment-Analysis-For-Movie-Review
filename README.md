# IMDb Sentiment Analysis

A machine learning project that predicts whether a given text sentiment is positive or negative. This project is based on the IMDb movie reviews dataset and includes a sentiment analysis model served through a Streamlit web application.

https://text-sentiment-analysis00.streamlit.app/

---

## 🎯 Project Overview
This project involves:
- **Dataset**: IMDb dataset for training and testing the sentiment analysis model.
- **Notebook**: A Jupyter Notebook used for preprocessing data, training the model, and evaluation.
- **Streamlit App**: An interactive web application where users can input text and receive sentiment predictions (positive/negative).
- **Saved Models**: Pre-trained machine learning models and tokenizers saved in `.pkl` format for inference.

---

## 🗂 Project Structure

```
├── IMDB dataset.csv              # Dataset containing IMDb movie reviews
├── IMDB_Sentiment_analysis.ipynb # Jupyter Notebook for model training and evaluation
├── app.py                        # Streamlit web app for sentiment prediction
├── model.pkl                     # Serialized trained sentiment analysis model
├── tokenizer.pkl                 # Tokenizer used in preprocessing
└── README.md                     # Documentation (you're reading this!)
```

---

## 🚀 How to Run the Project

### 1. **Clone the Repository**
```bash
git clone https://github.com/yourusername/Text-Sentiment-Analysis-Project.git
cd Text-Sentiment-Analysis-Project
```

### 2. **Install Required Dependencies**
Create a Python virtual environment and install the dependencies listed in `requirements.txt`.:

To install dependencies:
```bash
pip install -r requirements.txt
```

### 3. **Run the Streamlit App**
Start the Streamlit app to test the sentiment prediction:
```bash
streamlit run app.py
```
This will launch a local web page where you can input text and view the model's sentiment predictions.

---

## 📈 Workflow Details

### 1. **Dataset**
The dataset (`IMDB dataset.csv`) contains IMDb movie reviews, labeled as positive or negative. It serves as the basis for training and evaluation.

### 2. **Model Training**
The machine learning workflow for this project includes:
- **Data Preprocessing**: Tokenization, stop-word removal, HTML-tags removal and text vectorization.
- **Modeling**: A classification model built using libraries like Scikit-learn, TensorFlow.
- **Serialization**: The trained model and tokenizer are saved as `model.pkl` and `tokenizer.pkl` respectively, to be used for inference.

### 3. **Interactive Web App**
The Streamlit web app (`app.py`) allows users to:
1. Enter custom text inputs.
2. Receive predictions (positive/negative sentiment) powered by the trained model.

---

## 🛠 Technologies Used
- **Languages**: Python
- **Libraries**:
  - Machine Learning: Scikit-learn, TensorFlow/Keras
  - Data Handling: Pandas, NumPy
  - Visualization: Matplotlib, Seaborn
  - Web App: Streamlit
- **Tools**: Jupyter Notebook, Git, Virtual Environment

---

## 📊 Results
- **Accuracy**: 87.55% (calculated on the test dataset)
---

## 🌐 Web App Screenshots
Add some screenshots of your Streamlit app to make your project visually appealing.

![Screenshot 2025-01-06 153328](https://github.com/user-attachments/assets/361c9fe9-42e1-454c-918a-cf335dab938f)
