# Imports and Libraries
import pandas as pd
import numpy as np
import seaborn as sns
import tensorflow as tf
import keras
import pickle as pickle
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences



model = keras.models.load_model('sentModel.keras')
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)
    
def predictSentiment(text):
       # Tokenize and pad the input text
    text_sequence = tokenizer.texts_to_sequences([text])
    text_sequence = pad_sequences(text_sequence, maxlen=100)

    # Make a prediction using the trained model
    predicted_rating = model.predict(text_sequence)[0]
    print(predicted_rating)
    if np.argmax(predicted_rating) < 0.5:
        return 'Negative'
    else:
        return 'Positive'
def main():
    text_input = input("Enter review: ")
    predictedSent = predictSentiment(text_input)
    print(predictedSent)


main()