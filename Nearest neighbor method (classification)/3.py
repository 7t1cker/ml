import pandas as pd
import numpy as np

class MyKNNClf:
    def __init__(self, k=3, weight='uniform'):
        self.k = k
        self.weight = weight
        self.train_size = None
        self.X = None
        self.y = None

    def fit(self, X, y):
        self.X = X
        self.y = y
        self.train_size = X.shape
        return self.train_size

    def predict(self, X_test):
        predictions = np.empty(X_test.shape[0], dtype=np.int64)
        for i in range(X_test.shape[0]):
            distances = np.sqrt(np.sum((self.X - X_test.iloc[i])**2, axis=1))
            nearest_indices = distances.argsort()[:self.k]
            nearest_classes = self.y.iloc[nearest_indices]

            if self.weight == 'uniform':
                mode_class = nearest_classes.mode()[0]
            elif self.weight == 'rank':
                rank_weights = 1 / np.arange(1, self.k + 1) 
                weighted_counts = nearest_classes.value_counts(normalize=True) * rank_weights
                mode_class = weighted_counts.idxmax()
            elif self.weight == 'distance':
                distances_sum = np.sum(1 / (1 + distances[nearest_indices]))  
                inverse_distances = 1 / (1 + distances[nearest_indices])  
                weighted_counts = nearest_classes.value_counts(normalize=True) * inverse_distances
                mode_class = weighted_counts.idxmax()

            predictions[i] = mode_class

        return pd.Series(predictions)

    def predict_proba(self, X_test):
        probas = []
        for i in range(X_test.shape[0]):
            distances = np.sqrt(np.sum((self.X - X_test.iloc[i])**2, axis=1))
            nearest_indices = distances.argsort()[:self.k]
            nearest_classes = self.y.iloc[nearest_indices]

            if self.weight == 'uniform':
                prob_class_1 = nearest_classes.mean()
            elif self.weight == 'rank':
                rank_weights = 1 / np.arange(1, self.k + 1)  
                rank_weights_series = pd.Series(rank_weights, index=nearest_classes.unique())  
                weighted_counts = nearest_classes.value_counts(normalize=True) * rank_weights_series
                prob_class_1 = weighted_counts[1] 
            elif self.weight == 'distance':
                distances_sum = np.sum(1 / (1 + distances[nearest_indices]))  
                inverse_distances = 1 / (1 + distances[nearest_indices])  
                inverse_distances_series = pd.Series(inverse_distances, index=nearest_classes.unique())  
                prob_class_1 = weighted_counts[1] 

            probas.append(prob_class_1)




        return pd.Series(probas)