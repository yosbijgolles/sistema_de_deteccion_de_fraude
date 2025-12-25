import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class Log1p(BaseEstimator, TransformerMixin):

    def __init__(self, features=None):
        self.features = features or ['current_balance']

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_transformed = []

        for record in X:
            record_copy = record.copy()

            for feature in self.features:
                if feature in record_copy:
                    record_copy[feature] = np.log1p(record_copy[feature])

            X_transformed.append(record_copy)

        return X_transformed

    def get_feature_names_out(self, input_features=None):
        return input_features
