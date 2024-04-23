import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

class MyLineReg():
    def __init__(self, n_iter=100, learning_rate=0.1, weights=None, metric=None, reg=None, l1_coef=0, l2_coef=0):
        self.n_iter = n_iter
        self.learning_rate = learning_rate
        self.weights = weights
        self.metric = metric
        self.reg = reg
        self.l1_coef = l1_coef
        self.l2_coef = l2_coef
        self.best_score = None
        self.final_predictions = None

    def fit(self, X, y, verbose=False):
        X = pd.concat([pd.Series(1, index=X.index, name='bias'), X], axis=1)
        n_samples, n_features = X.shape
        
        if self.weights is None:
            self.weights = np.ones(n_features)
        
        if callable(self.learning_rate):
            dynamic_learning_rate = self.learning_rate
        else:
            dynamic_learning_rate = lambda iter: self.learning_rate
        
        for i in range(1, self.n_iter + 1):
            current_learning_rate = dynamic_learning_rate(i)
          
            y_pred = X.dot(self.weights)
            error = y_pred - y
            
            gradient = (2 / n_samples) * X.T.dot(error)
            
            if self.reg == 'l1':
                gradient[1:] += self.l1_coef * np.sign(self.weights[1:])
            elif self.reg == 'l2':
                gradient[1:] += 2 * self.l2_coef * self.weights[1:]
            elif self.reg == 'elasticnet':
                gradient[1:] += self.l1_coef * np.sign(self.weights[1:]) + 2 * self.l2_coef * self.weights[1:]
            
            self.weights -= current_learning_rate * gradient
            
            if self.metric is not None:
                self.final_predictions = X.dot(self.weights)
                metric_value = self.calculate_metric(y, self.final_predictions)
                if verbose:
                    print(f"{i} | {self.metric}: {metric_value:.10f} | Learning rate: {current_learning_rate:.10f}")
                self.best_score = metric_value

    def get_coef(self):
        return self.weights[1:]

    def predict(self, X):
        X = pd.concat([pd.Series(1, index=X.index, name='bias'), X], axis=1)
        return X.dot(self.weights)

    def calculate_metric(self, y_true, y_pred):
        if self.metric == 'mse':
            return mean_squared_error(y_true, y_pred)
        else:
            raise ValueError("Unsupported metric!")

    def get_best_score(self):
        return self.best_score


