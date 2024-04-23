import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

class MyLineReg():
    def __init__(self, n_iter=100, learning_rate=0.1, weights=None, metric=None):
        self.n_iter = n_iter
        self.learning_rate = learning_rate
        self.weights = weights
        self.metric = metric
        self.best_score = None
        self.final_predictions = None

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
            
            if self.metric is not None:
                current_loss = self.calculate_loss(X, y)
                self.final_predictions = X.dot(self.weights)
                metric_value = self.calculate_metric(y, self.final_predictions)
                if verbose:
                    print(f"{i + 1} | loss: {current_loss:.2f} | {self.metric}: {metric_value:.10f}")
                self.best_score = metric_value

    def get_coef(self):
        return self.weights[1:]

    def predict(self, X):
        X = pd.concat([pd.Series(1, index=X.index, name='bias'), X], axis=1)
        return X.dot(self.weights)

    def calculate_loss(self, X, y):
        y_pred = X.dot(self.weights)
        error = y_pred - y
        return np.mean(error ** 2)

    def calculate_metric(self, y_true, y_pred):
        if self.metric == 'mae':
            return mean_absolute_error(y_true, y_pred)
        elif self.metric == 'mse':
            return mean_squared_error(y_true, y_pred)
        elif self.metric == 'rmse':
            return np.sqrt(mean_squared_error(y_true, y_pred))
        elif self.metric == 'mape':
            return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        elif self.metric == 'r2':
            return r2_score(y_true, y_pred)

    def get_best_score(self):
        return self.best_score