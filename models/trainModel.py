# Imports and Libraries
import pandas as pd
import numpy as np
import tensorflow as tf
import math
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, GRU, BatchNormalization, Dense, Dropout, GlobalAveragePooling1D
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
import pickle as pickle
import re
import processData as procData
from sklearn.metrics import accuracy_score 

# This class is used to train the model

def setupModel(train_review):
    model = Sequential()
    model.add(Embedding(input_dim=50000, output_dim=100, input_length=200))
    model.add(GlobalAveragePooling1D())
    model.add(Dense(16, activation = 'relu'))
    model.add(Dense(1, activation='sigmoid')) # Sigmoid layer commonly used for binary classification problems, which is what we are dealing with
    model.build(input_shape=(None, 200))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.summary()
    return model

def trainModel():
    try:
        train_review, train_values, test_review, test_values = procData.preProcessData()
        model = setupModel(train_review)
        early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True) # here for testing, and to prevent overfitting but not needed
        trainedModel = model.fit(train_review, train_values, epochs=10, batch_size=32, validation_split=0.2, callbacks=[early_stopping])        
        loss, accuracy = model.evaluate(test_review, test_values, verbose=2)
        print(f'Test Accuracy: {accuracy} and loss {loss}')
        # Save the model and tokenizer if training is successful
        model.save('sentModel.keras')
        with open('model_history.pkl', 'wb') as file:
            pickle.dump(trainedModel.history, file)
    except Exception as e:
        print(f"An error has occurred: {e}")
trainModel()