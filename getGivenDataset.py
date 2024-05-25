# Imports and Libraries
import pandas as pd
import numpy as np
import re

#This class is for getting and extracting the data into a usable dataframe (may grab from web later, but for now we take it directly from dataset folder)
    # Parse and grab data from csv
def getDfCsv(labels, reviews):
    # Open and read the file
    with open(r'dataset\test.ft.txt', 'r', encoding='utf-8') as file:
        for line in file:
        # Split the line into label and review
            parts = line.split(' ', 1)  # Split on the first space
            if len(parts) == 2:  # Ensure the line is well-formed
                labels.append(parts[0])
                reviews.append(parts[1].strip())

def testerPrint(): # TO BE REMOVED LATER
    labels = []
    reviews = []
    getDfCsv(labels, reviews)
        # Create a DataFrame
    df = pd.DataFrame({
        'label': labels,
        'review': reviews   
    })
    # Display the first few rows of the DataFrame
    print(df.head())


testerPrint()