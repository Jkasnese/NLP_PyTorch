# TODO:
# Make path to file generic
# Make path to file be received from argument (CLI)

import os
import sys
import torch
import numpy as np
sys.path.append('../models')

zero_vec = []

def load_vocab_data(data_path='../datasets/ag_news_csv/train_processed.csv'):
    vocabulary = {}
    word_ID = 0
    data = []
    
    with open (data_path) as file:
        lines = [line.rstrip() for line in file]
        for line in lines:
            line = line.split("|")

            sentence = (' '.join(line[1:])).split()

            data.append((sentence, line[0]))

            for word in sentence:
                if word not in vocabulary:
                    vocabulary[word] = word_ID
                    word_ID += 1

    # Append to vocabulary in order to be divisible by 8 to speed GPU
    paddings = [" ", "  ", "   ", "    ", "     ", "      ", "       "]
    for i in range (8 - len(vocabulary) % 8):
        vocabulary[paddings[i]] = 0

    global zero_vec
    zero_vec = np.zeros(len(vocabulary))
    return vocabulary, data

def load_test_data(data_path: "absolute path to file" ='../datasets/ag_news_csv/test_processed.csv'):
    """ Function for loading test data into tuple list
    Same thing as function above, but no dictionary.
    Another function to avoid comparisons in previous function and degrage performance
    (is it worth it?)
    """
    data = []

    with open (data_path) as file:
        lines = [line.rstrip() for line in file]
        for line in lines:
            line = line.split("|")
            sentence = (' '.join(line[1:])).split()
            data.append((sentence, line[0]))
        return data


# TODO: 
# Use yield
# make a copy of np.zeros once to prevent from making every call, just making a copy of this vector in next call
# Change load vocab to store text as indexes to the words. So you only have to add the number here
def make_bow_vector(sentence : "smallest text unit to classify", vocabulary : "dictionary"):
    """ Makes bag of words tensor given a sentence and a vocabulary """
    cp_vec = zero_vec
    for word in sentence:
        if word in vocabulary:
            cp_vec[vocabulary[word]] += 1
    return cp_vec # Shapes the tensor to 1 dimension (since its linear). Infers other dimension


