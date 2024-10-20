import pandas as pd
from sklearn.base import (BaseEstimator, ClassNamePrefixFeaturesOutMixin,
                          TransformerMixin)


class ZeroTransformer(BaseEstimator,TransformerMixin,ClassNamePrefixFeaturesOutMixin):

    def fit(self, X: pd.DataFrame, y=None):
        return self
    
    def transform(self, X: pd.DataFrame, y = None):
        return pd.DataFrame()
