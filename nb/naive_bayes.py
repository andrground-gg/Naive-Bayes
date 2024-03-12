from math import exp
from math import pi
from math import sqrt
import numpy as np

class NaiveBayes:
    def __init__(self):
        self.__classes = None
        self.__class_priors = None
        self.__means = {}
        self.__variances = {}

    def fit(self, X, y):
        self.__classes = np.unique(y)
        self.__class_priors = [len(X[y == c]) / len(X) for c in self.__classes]
        for c in self.__classes:
            X_c = X[y == c]
            self.__means[c] = X_c.mean(axis=0)
            self.__variances[c] = X_c.var(axis=0)

    def predict(self, X):
        pass

    # def __mean(self, a):
    #     return sum(a) / len(a)
    
    # def __variance(self, a):
    #     mean = self.__mean(a)
    #     return sum([(x - mean) ** 2 for x in a]) / len(a)
    
    def __calculate_likelihood(self, x, mean, variance):
        exponent = exp(-(x - mean) ** 2 / (2 * variance))
        return (1 / (sqrt(2 * pi * variance))) * exponent
    
    def __calculate_class_probabilities(self, x):
        pass
    
    


