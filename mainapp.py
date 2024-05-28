# Imports and Libraries
import pandas as pd
import numpy as np
import tensorflow as tf
import keras
import pickle as pickle
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)
# Import model in
model = keras.models.load_model('sentModel.keras')
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

def preprocessText(text):
    # Tokenize and pad the input text
    text_sequence = tokenizer.texts_to_sequences([text])    
    pad_sequence = pad_sequences(text_sequence, maxlen=200, truncating='post')
    return pad_sequence

def analyzeText(text):
    processedText = preprocessText(text)
    prediction = model.predict(processedText)
    if prediction >= 0.5:
        return 'Positive'
    else:
        return 'Negative'

@app.route('/analyze', methods=['POST'])
def analyzeManualSentiment():
    data = request.json
    text = data.get('text')
    # Your sentiment analysis logic here
    sentiment = analyzeText(text)  # Replace with your actual function
    print(f'Sentiment was: {sentiment}')
    return jsonify({'sentiment': sentiment})

def analyzeFileSentiment(file_path):
    pass


@app.route('/uploadFile', methods=['POST'])
def analyzeFileSentimentInfo():
    pass

if __name__ == '__main__':
    app.run(debug=True)
