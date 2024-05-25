This project will be completed in the following steps
1. Data collection and Cleaning
    a. First need to parse and collect different datasets to feed into model first to train it
    b. Considerable time will be spent breaking things up so that it can be used in dataframes
2. Pre-processing data
    a. Data must be pre-processed so that the datasets will have certain words/keywords associated with values
    b. Cleaning, tokenizing
3. Train model using Logistic Regression Model Architecture or Sequential neural network because dataset is larger
    a. Logistic Regression is great for models that will quantify things as either 0 or 1 boolean values so for a sentiment analyzer it will be perfect for reviews
    b. However it is simplistic
        (steps are basically: preprocess data, get word embeddings, define the architecture of the model, train the model, test the model)
4. Make adjustments to model based on metric information gathered to tune it 
5. Create metrics to be sent to front end
6. Display information in graphs for user
7. Allow for any comment to be inputted to saved model and then get a response back if it is positive or negative based on models decision