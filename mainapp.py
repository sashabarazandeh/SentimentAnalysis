# Imports and Libraries
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from collections import Counter
import keras
import pickle as pickle
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from flask import Flask, request, jsonify
from flask_cors import CORS
from graphSentiment import plotSentimentBar, plotSentimentPie

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
    sentimentList = []
    positiveNumber = 0
    negativeNumber = 0
    with open(file_path, 'r', encoding='UTF-8') as reviews:
        for review in reviews:
            text = review.strip()
            if text:
                sentiment = analyzeText(text)
                if sentiment == 'Positive':
                    positiveNumber += 1
                else:
                    negativeNumber += 1
                sentimentList.append(sentiment)
    totalReviews = positiveNumber + negativeNumber
    positivePercent = 100*(positiveNumber/totalReviews)
    negativePercent = 100*(negativeNumber/totalReviews)
    truncatePos = int(positivePercent)
    truncateNeg = int(negativePercent)
    positivePercent = (truncatePos/100)
    negativePercent = (truncateNeg/100)
    plotSentimentBar(positiveNumber, negativeNumber)
    plotSentimentPie(positiveNumber, negativeNumber)
    barPng = 'Bar_Graph.png'
    piePng = 'Pie_Chart.png'
    return sentimentList, positivePercent*100, negativePercent*100, barPng, piePng


@app.route('/upload', methods=['POST'])
def analyzeFileSentimentInfo():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if file:
        file_path = 'uploaded_file.txt'
        file.save(file_path)
        sentimentList, positivePercent, negativePercent, barPng, piePng = analyzeFileSentiment(file_path)
        return jsonify({'sentiments': sentimentList}, {'positive': positivePercent}, {'negative': negativePercent}, {'barPng': barPng}, {'piePng': piePng})

if __name__ == '__main__':
    app.run(debug=True)
