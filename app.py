from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.tokenize import word_tokenize
import nltk
import re

# Download NLTK resources
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')

# Initialize Flask app
app = Flask(_name_)
CORS(app)

# Load the saved models
base_path = 'C:\\Users\\Pavithra G R\\Documents\\NewProject\\FrontendApp\\backend\\'
tfidf = joblib.load(base_path + 'tfidf_pipeline_optimized.pkl')
xgb_model = joblib.load(base_path + 'xgb_category_model_optimized.pkl')
label_encoder = joblib.load(base_path + 'label_encoder.pkl')

# Preprocessing function (from your training code)
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    stemmer = SnowballStemmer("english")
    tokens = [stemmer.stem(word) for word in tokens]
    return ' '.join(tokens)

# Prediction function
def predict_category(text):
    processed_text = preprocess_text(text)
    print(f"Processed text: {processed_text}")
    text_tfidf = tfidf.transform([processed_text]).toarray()
    print(f"TF-IDF shape: {text_tfidf.shape}")
    prediction = xgb_model.predict(text_tfidf)
    print(f"Raw prediction: {prediction}")
    predicted_label = label_encoder.inverse_transform(prediction)[0]
    print(f"Predicted label: {predicted_label}")
    return predicted_label

@app.route('/')
def home():
    return jsonify({'message': 'Welcome to the Complaint Prediction API'})

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        text = data.get('text', '')
        if not text:
            return jsonify({'error': 'No text provided'}), 400
        predicted_category = predict_category(text)
        return jsonify({'prediction': predicted_category})
    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({'error': str(e)}), 500

if _name_ == '_main_':
    app.run(host='127.0.0.1', port=5000, debug=True)