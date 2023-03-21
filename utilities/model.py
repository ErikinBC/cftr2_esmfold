"""
Official model to be used by 9_predict_y.py
"""

# Load external modules
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.experimental import enable_iterative_imputer as _
from sklearn.impute import IterativeImputer
# Load internal modules
from parameters import seed, di_mdl_class, mdl_class


class mdl(BaseEstimator, RegressorMixin):
    def __init__(self) -> None:
        # Initialize algorithms and encoders
        self.algorithm = mdl_class(**di_mdl_class)
        self.enc_x = StandardScaler()
        self.enc_y = StandardScaler()
        # Iterative imputer is hard-coded except for the seed
        self.enc_imp = IterativeImputer(verbose=0, random_state=seed, max_iter=1000, imputation_order='random')


    def fit(self, X:pd.DataFrame, y:pd.DataFrame) -> None:
        assert isinstance(X, pd.DataFrame)
        assert isinstance(y, pd.DataFrame)
        self.cn_X = X.columns
        self.cn_y = y.columns
        # Learning scaling
        self.enc_x.fit(X)
        self.enc_y.fit(y)
        self.enc_imp.fit(y)
        # impute missing Y's
        y_imp = y.mask(y.isnull(), self.enc_imp.transform(y))
        # scale X/y
        Xtil = self.enc_x.transform(X)
        ytil = self.enc_y.transform(y_imp)
        # fit learning algorithm
        self.algorithm.fit(Xtil, ytil)


    def predict(self, X:pd.DataFrame) -> np.ndarray:
        # Input checks
        assert isinstance(X, pd.DataFrame)
        assert (X.columns == self.cn_X).all()
        # transform the X-space
        Xtil = self.enc_x.transform(X)
        # make prediction
        ytil = self.algorithm.predict(Xtil)
        # put prediction back to original scale
        y = self.enc_y.inverse_transform(ytil)
        return y
