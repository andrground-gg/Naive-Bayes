import numpy as np
import pandas as pd

class NaiveBayes:
    def __init__(self):
        self.__classes = None
        self.__class_priors = None
        self.__classes_amounts = None
        self.__data = None

    def fit(self, X, y):
        self.__classes = np.unique(y)
        self.__class_priors = [len(X[y == c]) / len(X) for c in self.__classes]
        self.__classes_amounts = [len(X[y == c]) for c in self.__classes]
        dict = {}
        for col in X.columns:
            dict[col] = {}
            for c in np.unique(X[col]):
                dict[col][c] = tuple([len(X[(X[col] == c) & (y == 0)]), len(X[(X[col] == c) & (y == 1)])])
        
        self.__data = dict
        
    def predict(self, X):
        predictions = []
        for index, row in X.iterrows():
            neg_prob, pos_prob = self.__class_priors[0], self.__class_priors[1]
            for key in self.__data.keys():
                neg_prob *= self.__data[key][X[key][index]][0] / self.__classes_amounts[0] if X[key][index] in self.__data[key] and self.__data[key][X[key][index]][0] != 0 else 1 / len(X)
                pos_prob *= self.__data[key][X[key][index]][1] / self.__classes_amounts[1] if X[key][index] in self.__data[key] and self.__data[key][X[key][index]][1] != 0 else 1 / len(X)

            predictions.append(0 if neg_prob > pos_prob else 1)

        return np.array(predictions, dtype=np.int64)

    def __calculate_likelihood(self, x, mean, variance):
        pass

    def __calculate_class_probabilities(self, x):
        pass
        