import numpy as np
import pandas as pd
class MyLineReg():
    def __init__(self, n_iter=100, learning_rate=0.1, weights=None):
        self.n_iter = n_iter
        self.learning_rate = learning_rate
        self.weights = weights

    def fit(self, X, y, verbose=False):
        X = pd.concat([pd.Series(1, index=X.index, name='bias'), X], axis=1)
        n_samples, n_features = X.shape
        
        if self.weights is None:
            self.weights = np.ones(n_features)
        
        for i in range(self.n_iter):
            y_pred = X.dot(self.weights)
            error = y_pred - y
            gradient = (2 / n_samples) * X.T.dot(error)
            self.weights -= self.learning_rate * gradient
            
            if verbose and (i + 1) % verbose == 0:
                current_loss = self.calculate_loss(X, y)
                print(f"{i + 1} | loss: {current_loss:.2f}")

    def get_coef(self):
        return self.weights[1:]

    def calculate_loss(self, X, y):
        y_pred = X.dot(self.weights)
        error = y_pred - y
        return np.mean(error ** 2)

    def predict(self, X):
        X = pd.concat([pd.Series(1, index=X.index, name='bias'), X], axis=1)
        return X.dot(self.weights)