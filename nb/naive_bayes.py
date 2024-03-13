import numpy as np

class NaiveBayes:
    def __init__(self, log=False):
        self.__classes = None
        self.__class_priors = None
        self.__means = {}
        self.__variances = {}
        self.__log = log

    def fit(self, X, y):
        self.__classes = np.unique(y)
        self.__class_priors = [len(X[y == c]) / len(X) for c in self.__classes]
        for c in self.__classes:
            X_c = X[y == c]
            self.__means[c] = X_c.mean(axis=0)
            self.__variances[c] = X_c.var(axis=0)

        if (self.__log):
            print('Classes: ', self.__classes)
            print('Class priors: ', self.__class_priors)
            print('Means: ', self.__means)
            print('Variances: ', self.__variances)

    def predict(self, X):
        probabilities = self.__calculate_class_probabilities(X)
        predictions = probabilities[1] > probabilities[0]
        return predictions.astype('int')
    
    def __calculate_likelihood(self, x, mean, variance):
        exponent = np.exp(-(x - mean) ** 2 / (2 * variance))
        return (1 / (np.sqrt(2 * np.pi * variance))) * exponent
    
    def __calculate_class_probabilities(self, x):
        probabilities = {}
        for c in self.__classes:
            likelihood = np.prod(self.__calculate_likelihood(x, self.__means[c], self.__variances[c]), axis=1)
            probabilities[c] = self.__class_priors[c] * likelihood

        if (self.__log):
            print('Class probabilities: ', probabilities)

        return probabilities
        

    
    


