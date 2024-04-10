import numpy as np
import pandas as pd

class NaiveBayes:
    def __init__(self):
        # Inicializuje objekt triedy NaiveBayes.

        # Atribúty:
        # - __classes: Pole s unikátnymi triedami cieľového atribútu.
        # - __class_priors: Pole s apriórnymi pravdepodobnosťami tried.
        # - __data: Slovník, ktorý uchováva informácie o dátach pre každý atribút.
        self.__classes = None
        self.__class_priors = None
        self.__data = {}

    def fit(self, X, y):
        # Trénovanie modelu
        self.__classes = np.unique(y)
        self.__class_priors = [len(X[y == c]) / len(X) for c in self.__classes]

        for col in X.columns:
            self.__data[col] = {}
            for c in np.unique(X[col]):
                self.__data[col][c] = tuple([len(X[(X[col] == c) & (y == 0)]), len(X[(X[col] == c) & (y == 1)])])
        
    def predict(self, X):
        # Predikuje triedy pre zadané dáta.
        predictions = []
        probabilities = self.__calculate_class_probabilities(X)
        for probs in probabilities:
            predictions.append(0 if probs[0] > probs[1] else 1)

        return np.array(predictions, dtype=np.int64)

    def __calculate_likelihoods(self, X, key, index):
        # Vypočíta pravdepodobnosti pre zadaný atribút a hodnotu v ňom.

        # Parametre:
        # - X: Dátový rámec s príznakmi.
        # - key: Kľúč (názov atribútu).
        # - index: Index riadka v dátovom rámci.

        # Výstup:
        # Tuple s pravdepodobnosťami pre pozitívne a negatívne triedy.
        pos_probability = 1 / len(X)
        neg_probability = 1 / len(X)

        if (X[key][index] in self.__data[key]):
            if (self.__data[key][X[key][index]][0] != 0):
                pos_probability = self.__data[key][X[key][index]][0]
            if (self.__data[key][X[key][index]][1] != 0):
                neg_probability = self.__data[key][X[key][index]][1]

        return (neg_probability, pos_probability)

    def __calculate_class_probabilities(self, X):
        # Vypočíta pravdepodobnosti pre každý riadok vstupných dát.

        # Výstup:
        # Pole tuple s pravdepodobnosťami pre pozitívnu a negatívnu triedu pre každý riadok.
        probabilities = []
        for index, _ in X.iterrows():
            neg_prob, pos_prob = self.__class_priors[0], self.__class_priors[1]
            for key in self.__data.keys():
                likelihood_neg, likelihood_pos = self.__calculate_likelihoods(X, key, index)
                neg_prob *= likelihood_pos 
                pos_prob *= likelihood_neg
                
            probabilities.append((neg_prob, pos_prob))

        return probabilities
        