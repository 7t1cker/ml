import pandas as pd

class MyKNNClf:
    def __init__(self, k=3):
        self.k = k
        self.train_size = None 

    def fit(self, X, y):
        self.X = X  
        self.y = y  
        self.train_size = X.shape 
        return self.train_size