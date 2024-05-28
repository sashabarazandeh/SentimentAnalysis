# Imports and Libraries
import pandas as pd
import numpy as np
import tensorflow as tf
import keras
import pickle as pickle
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences



model = keras.models.load_model('sentModel.keras')
model.summary()
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)
with open('model_history.pkl', 'rb') as handle:
    modHist = pickle.load(handle)

def predictSentiment(text):
    # Tokenize and pad the input text
    text_sequence = tokenizer.texts_to_sequences([text])
    print(f'Text Sequence: {text_sequence}')
    
    text_sequence = pad_sequences(text_sequence, maxlen=200, truncating='post')
    print(f'Padded Sequence: {text_sequence}')

    # Make a prediction using the trained model
    predicted_rating = model.predict(text_sequence)
    print('Predicted Rating: ', predicted_rating)
    if predicted_rating >= 0.5:
        return 'Positive'
    else:
        return 'Negative'
def main():
    text_input = input("Enter review: ")
   # text_input = "I love this battery, it works really well and lasts for a long time. It seems to be a good purchase"
    predictedSent = predictSentiment(text_input)
    print(f'Your sentiment is: {predictedSent}')
    print(f"Model History: Training Accuracy: {modHist['accuracy']}, \n Validation Accuracy: {modHist['val_accuracy']}, \n Training Loss: {modHist['loss']}, \n Validation Loss: {modHist['val_loss']}")

main()