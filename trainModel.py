# Imports and Libraries
import pandas as pd
import numpy as np
import tensorflow as tf
import math
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
import pickle as pickle
import re
import processData as procData
from sklearn.metrics import accuracy_score 

# This class is used to train the model

def setupModel(train_review):
    model = Sequential()
    model.add(Embedding(input_dim=25000, output_dim=16, input_length=200))
    model.add(Bidirectional(LSTM(32, return_sequences=True)))
    model.add(Dropout(0.5))
    model.add(Bidirectional(LSTM(32)))
    model.add(Dropout(0.5))
    model.add(Dense(16, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid')) # Sigmoid layer commonly used for binary classification problems, which is what we are dealing with
    model.build(input_shape=(None, 200))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.summary()
    return model

def trainModel():
    try:
        train_review, train_values, test_review, test_values = procData.preProcessData()
        train_review, val_review, train_values, val_values = train_test_split(train_review, train_values, test_size=0.2, random_state=42)
        model = setupModel(train_review)
        early_stopping = EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)
        trainedModel = model.fit(train_review, train_values, epochs=12, batch_size=32, validation_split=0.2)        
        loss, accuracy = model.evaluate(test_review, test_values, verbose=2)
        print(f'Test Accuracy: {accuracy} and loss {loss}')
        # Save the model and tokenizer if training is successful
        model.save('sentModel.keras')
    except Exception as e:
        print(f"An error has occurred: {e}")
trainModel()