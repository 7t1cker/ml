class MyForestReg:
    def __init__(self, n_estimators=10, max_features=0.5, max_samples=0.5, random_state=42,
                 max_depth=5, min_samples_split=2, max_leafs=20, bins=16):
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.max_samples = max_samples
        self.random_state = random_state
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_leafs = max_leafs
        self.bins = bins

    def __str__(self):
        return f"MyForestReg class: n_estimators={self.n_estimators}, " \
               f"max_features={self.max_features}, max_samples={self.max_samples}, " \
               f"max_depth={self.max_depth}, min_samples_split={self.min_samples_split}, " \
               f"max_leafs={self.max_leafs}, bins={self.bins}, random_state={self.random_state}"