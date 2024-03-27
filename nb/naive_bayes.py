import numpy as np
import pandas as pd

class NaiveBayes:
    def __init__(self):
        self.__classes = None
        self.__class_priors = None
        self.__data = {}

    def fit(self, X, y):
        self.__classes = np.unique(y)
        self.__class_priors = [len(X[y == c]) / len(X) for c in self.__classes]

        for col in X.columns:
            self.__data[col] = {}
            for c in np.unique(X[col]):
                self.__data[col][c] = tuple([len(X[(X[col] == c) & (y == 0)]), len(X[(X[col] == c) & (y == 1)])])
        
    def predict(self, X):
        predictions = []
        probabilities = self.__calculate_class_probabilities(X)
        for probs in probabilities:
            predictions.append(0 if probs[0] > probs[1] else 1)

        return np.array(predictions, dtype=np.int64)

    def __calculate_likelihoods(self, X, key, index):
        pos_probability = 1 / len(X)
        neg_probability = 1 / len(X)

        if (X[key][index] in self.__data[key]):
            if (self.__data[key][X[key][index]][0] != 0):
                pos_probability = self.__data[key][X[key][index]][0]
            if (self.__data[key][X[key][index]][1] != 0):
                neg_probability = self.__data[key][X[key][index]][1]

        return (neg_probability, pos_probability)

    def __calculate_class_probabilities(self, X):
        probabilities = []
        for index, _ in X.iterrows():
            neg_prob, pos_prob = self.__class_priors[0], self.__class_priors[1]
            for key in self.__data.keys():
                likelihood_neg, likelihood_pos = self.__calculate_likelihoods(X, key, index)
                neg_prob *= likelihood_pos 
                pos_prob *= likelihood_neg
                
            probabilities.append((neg_prob, pos_prob))

        return probabilities
        