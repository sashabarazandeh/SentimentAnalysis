# Imports and Libraries
import pandas as pd
import numpy as np
import tensorflow as tf
import math
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense, Dropout
import pickle as pickle
import re
import processData as procData
from sklearn.metrics import accuracy_score 

# This class is used to train the model
def setupModel(train_review, train_values, test_review, test_values):
    model = Sequential()
    model.add(Embedding(7000, 100, input_length=100))
    model.add(Bidirectional(LSTM(64, return_sequences=True)))
    model.add(Dropout(0.5))
    model.add(Bidirectional(LSTM(64)))
    model.add(Dropout(0.5))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid')) # Sigmoid layer commonly used for binary classification problems, which is what we are dealing with
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.summary()
    return model


def trainModel():
    train_review, train_values, test_review, test_values, tokenizer = procData.preProcessData()
    model = setupModel(train_review, train_values, test_review, test_values)
    trainedModel = model.fit(train_review, train_values, epochs=10, batch_size=32, validation_data=(test_review, test_values), verbose=2)
    loss, accuracy = model.evaluate(test_review, test_values, verbose=2)
    print(f'Test Accuracy: {accuracy} and loss {loss}')
    model.save('sentModel.h5')
    with open('tokenizer.pickle', 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
trainModel()