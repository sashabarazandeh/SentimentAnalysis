# Imports and Libraries
import pandas as pd
import numpy as np
import tensorflow as tf
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import re


#initialize tokenizer
tokenizer = Tokenizer()
# This class is for pre-processing of data for the model to be able to train itself to understand what constitues positive, negative and neutral responses
def tokenizeAndPadData(value, review):
    value = value.apply(lambda x: 1 if x == '2'
                                                  else 0 )
    # Convert train_review into values of integers using tokenizer and setup word index
    tokenizer = Tokenizer(num_words=10000, oov_token='<OOV>')
    tokenizer.fit_on_texts(review) # break up words into tokens and create word index
    wordIndex = tokenizer.word_index #set word index
    sequences = tokenizer.texts_to_sequences(review)
    padded_sequences = pad_sequences(sequences, maxlen= 250)
    sentimentValues = pd.get_dummies(value).values
    return padded_sequences, sentimentValues

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
    test_value, test_review = testReviews['Sentiment'], testReviews['Review']
    #train_review = train_review.lower()
    #test_review = test_review.lower()
    padTrainReview, padTrainValue = tokenizeAndPadData(train_value, train_review)
    padTestReview, padTestValue = tokenizeAndPadData(test_value, test_review)
    return padTrainReview, padTrainValue, padTestReview, padTestValue, tokenizer

def preProcessData():
    return collectData()
preProcessData()

