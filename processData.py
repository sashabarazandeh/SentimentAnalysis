# Imports and Libraries
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import Tokenizer
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from keras import pad_sequences
from keras import Sequential
from keras import Embedding, Bidirectional, LSTM, Dense, Dropout
import pickle as pickle
import re

# This class is for pre-processing of data for the model to be able to train itself to understand what constitues positive, negative and neutral responses
def collectData():
    productReviews = pd.read_csv("dataset/train.csv") 
    testReviews = pd.read_csv("dataset/test.csv") 
    # Columns arent named initially for this dataset, so name them appropriately
    productReviews.columns = ['Sentiment', 'Title', 'Review']
    testReviews.columns = ['Sentiment', 'Title', 'Review']
    
    # maybe as a side note, I can combine titles and reviews into one because the test can still be read/used?

    # Now that we have two seperate dataframes containing the test.csv and the train.csv, we want to split them up into the actual text and the sentiment values
    # We only care about the sentiment and the actual review, the title isn't as important as it is sort of like a 'second shorter' review
    train_value, train_review = productReviews['Sentiment'], productReviews['Review']
    test_Value, test_review = testReviews['Sentiment'], testReviews['Review']

    
collectData()