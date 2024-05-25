# Imports and Libraries
import pandas as pd
import numpy as np
import seaborn as sns
import tensorflow as tf
import math
from flask import Flask
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction import IfidVectorizer
from sklearn.model_selection import train_test_split
import re
from sklearn.metrics import accuracy_score 

# This class is used to train the model