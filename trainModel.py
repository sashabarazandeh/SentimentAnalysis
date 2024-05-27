# Imports and Libraries
import pandas as pd
import numpy as np
import tensorflow as tf
import math
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import pickle as pickle
import re
import processData as procData
from sklearn.metrics import accuracy_score 

# This class is used to train the model
def setupModel(train_review, train_values, test_review, test_values):
    model = Sequential()
    model.add(Embedding(input_dim=25000, output_dim=100, input_length=100))
    model.add(Bidirectional(LSTM(32, return_sequences=True)))
    model.add(Dropout(0.5))
    model.add(Bidirectional(LSTM(32)))
    model.add(Dropout(0.5))
    model.add(Dense(16, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid')) # Sigmoid layer commonly used for binary classification problems, which is what we are dealing with
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def trainModel():
    try:
        train_review, train_values, test_review, test_values, tokenizer = procData.preProcessData()
        model = setupModel(train_review, train_values, test_review, test_values)
        model.summary()
        # Callbacks for monitoring and early stopping
        early_stopping = EarlyStopping(monitor='val_loss', patience=3)
        model_checkpoint = ModelCheckpoint('best_model.keras', save_best_only=True, monitor='val_loss')
        
        trainedModel = model.fit(train_review, train_values, epochs=10, batch_size=32, validation_data=(test_review, test_values), verbose=2)
        loss, accuracy = model.evaluate(test_review, test_values, verbose=2)
        print(f'Test Accuracy: {accuracy} and loss {loss}')
        # Save the model and tokenizer if training is successful
        model.save('sentModel.keras')
        with open('tokenizer.pickle', 'wb') as handle:
            pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception as e:
        print(f"An error occurred: {e}")
trainModel()