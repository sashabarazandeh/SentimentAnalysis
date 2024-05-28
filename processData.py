# Imports and Libraries
import pandas as pd
import numpy as np
import tensorflow as tf
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from collections import Counter
import pickle as pickle
import re
import string


lemmatizer = WordNetLemmatizer()
stopWords = set(stopwords.words('english'))
# This class is for pre-processing of data for the model to be able to train itself to understand what constitues positive, negative and neutral responses
def cleanData(review):
    words = review.split()
    cleanedWords = []
    for text in words:
        #Remove digits
        text = re.sub(r'\d+', '', text)
        #Remove special characters
        text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
        #Remove stopwords and use lemmatizer to lemmatize similar words and lowercase
        if text not in stopWords:
            text = lemmatizer.lemmatize(text)
            text = text.lower()
            cleanedWords.append(text)
    cleanedReview = ' '.join(cleanedWords)
    return cleanedReview

def tokenizeAndPadData(value, review, trainOrTest):
    binaryValues = value.apply(lambda x: 0 if x == 1
                               else 1)
    
    if trainOrTest == 'TRAIN':
    # Convert train_review into values of integers using tokenizer and setup word index
        tokenizer = Tokenizer(num_words=10000, oov_token='<OOV>')
        tokenizer.fit_on_texts(review) # break up words into tokens and create word index
        wordIndex = tokenizer.word_index #set word index
        print(f'word index max {max(wordIndex.values())}')
        sequences = tokenizer.texts_to_sequences(review)
        padded_sequences = pad_sequences(sequences, maxlen= 150, truncating='post')
        sentimentValues = binaryValues.values
        #print(padded_sequences, sentimentValues[:20])
        with open('tokenizer.pickle', 'wb') as handle:
            pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        with open('tokenizer.pickle', 'rb') as handle:
            tokenizer = pickle.load(handle)
        sequences = tokenizer.texts_to_sequences(review)
        padded_sequences = pad_sequences(sequences, maxlen= 200, truncating='post')
        sentimentValues = binaryValues.values
        #print(padded_sequences, sentimentValues[:20])
    return padded_sequences, sentimentValues

def collectData():
    productReviews = pd.read_csv("dataset/train.csv", nrows = 80000) 
    testReviews = pd.read_csv("dataset/test.csv", nrows = 30000) 
          
    # Columns arent named initially for this dataset, so name them appropriately
    productReviews.columns = ['Sentiment', 'Title', 'Review']
    testReviews.columns = ['Sentiment', 'Title', 'Review']


    # maybe as a side note, I can combine titles and reviews into one because the test can still be read/used?
    # Now that we have two seperate dataframes containing the test.csv and the train.csv, we want to split them up into the actual text and the sentiment values
    # We only care about the sentiment and the actual review, the title isn't as important as it is sort of like a 'second shorter' review
    train_value, train_review = productReviews['Sentiment'], productReviews['Review']
    test_value, test_review = testReviews['Sentiment'], testReviews['Review']

    # Clean text, lemmatize it and remove stopwords to improve model accuracy
    train_review = train_review.apply(lambda x: cleanData(str(x)))
    test_review = test_review.apply(lambda x: cleanData(str(x)))

    padTrainReview, padTrainValue = tokenizeAndPadData(train_value, train_review, 'TRAIN')
    padTestReview, padTestValue = tokenizeAndPadData(test_value, test_review, 'TEST')    

    return padTrainReview, padTrainValue, padTestReview, padTestValue

def preProcessData():
    return collectData()
preProcessData()

