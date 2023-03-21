"""
Utility methods for statistical functions
"""

# Load modules
import numpy as np
import pandas as pd
from typing import Callable
from sklearn.metrics import r2_score
from sklearn.base import BaseEstimator
from scipy.stats import kendalltau, spearmanr
from sklearn.preprocessing import StandardScaler
from statsmodels.nonparametric.kernel_regression import KernelReg
# Internal modules
from utilities.utils import str2list, cvec, merge_pairwise_dfs


def gauss_kernel(z:np.ndarray) -> np.ndarray:
    """Function for the Gaussian kernel"""
    return np.exp(-z**2 / 2) / np.sqrt(2*np.pi)


class NadarayaWatson(BaseEstimator):
    def __init__(self, **kwargs) -> None:
        """
        Implements an Sklearn-like Nadarya-Watson estimator. Any arguments can be passed on KernelReg(...) class with **kwargs
        """
        # Fits a continuous (c) regression with a local constant (lc) - NW-estimator, and determines the bandwidth using LOO-CV (cv_ls)
        self.di_KReg = {'var_type':'c','reg_type':'lc','bw':'cv_ls', 'ckertype':'gaussian'}
        if len(kwargs) > 0:
            for k in kwargs:
                if k in self.di_KReg:
                    self.di_KReg[k] = kwargs[k]
    

    def fit(self, X:np.ndarray, y=np.ndarray) -> None:
        assert len(X) == len(y), 'X and y need to be the same length'
        # Ensure with have 2-dim data
        X = cvec(X)
        self.k = X.shape[1]
        assert self.k == 1, 'Expected univariate input X'
        # Have scaler for the X-data
        self.enc_X = StandardScaler()
        self.enc_X.fit(X)
        X_til = self.enc_X.transform(X)
        # Fit the N-W estimator
        self.nw = KernelReg(endog=y, exog=X_til, **self.di_KReg)
        self.h = self.nw.bw  # extract bandwith
        # Calculate the training error
        self.var_yhat = (y - self.nw.fit(X_til)[0])**2

    def predict(self, X:np.ndarray) -> pd.DataFrame:
        # Input checks
        X_til = self.enc_X.transform(cvec(X))
        assert X_til.shape[1] == self.k
        # Get the predicted mean
        yhat = self.nw.fit(X_til)[0]
        # Get the weights of the new observations
        w_til = gauss_kernel((X_til - self.nw.exog.T) / self.h)
        sigma_hat = np.sqrt((w_til*self.var_yhat).sum(1) / w_til.sum(1))
        # Return predicted mean and se as a DF
        res = pd.DataFrame({'yhat':yhat, 'se':sigma_hat})
        return res
        
        
def get_perf_msrs(df:pd.DataFrame, cn_gg:str or list, cn_y:str, cn_yhat:str) -> pd.DataFrame:
    """Get the R2, spearman, and kendal measures of correlation"""
    assert isinstance(df, pd.DataFrame)
    cn_gg = str2list(cn_gg)
    res = df.groupby(cn_gg).apply(lambda x: pd.DataFrame({'r2':r2_score(x[cn_y], x[cn_yhat]), 'tau':kendalltau(x[cn_y].values, x[cn_yhat].values)[0], 'rho':spearmanr(x[cn_y].values, x[cn_yhat].values)[0]},index=[0]))
    res = res.reset_index().drop(columns=f'level_{len(cn_gg)}')
    res = res.melt(cn_gg,var_name='msr')
    return res


def perf_btw_mats(df1:pd.DataFrame, df2: pd.DataFrame) -> pd.DataFrame:
    """
    For two matrices of the same shape/columns, do a column-wise correlation and r-squared (note df1 treatd as the "actual" for R2 calculation)
    """
    # Merge pairwise data dropping NAs
    res = merge_pairwise_dfs(df1, df2)
    # Get column-wise correlation and R2
    res = get_perf_msrs(res, 'cn', 'value_x', 'value_y').pivot('cn','msr','value')
    return res


def bootstrap_function(df:pd.DataFrame, function:Callable, cn_val:str, cn_gg:str or list, n_boot:int=100, alpha:float=0.05, di_args:dict={}) -> pd.DataFrame:
    """
    A generic wrapper for getting the bootstrapped confidence intervals for some "function"

    Parameters
    ----------
    df:             DataFrame to be passed to callable
    function:       Function to calculate the statistics of interest (must return a pd.DataFrame)
    cn_val:         The column name of the statistic that is returned by function
    cn_gg:          Whether the bootstrapped should be grouped by a set of columns
    nboot:          Number of bootstrap iterations
    alpha:          Type-I error rate
    di_args:        **kwargs to be passed to function(df, **kwargs)

    Returns
    -------
    A DataFrame matching the columns of function() along with lb/ub/se/is_sig from the bootstrap
    """
    # (i) Input checks
    assert isinstance(df, pd.DataFrame), 'df needs to be a dataframe'
    assert callable(function), 'function needs to be callable'
    assert isinstance(cn_val, str)
    cn_gg = str2list(cn_gg)
    
    # (ii) Get the baseline value
    res0 = function(df, **di_args)
    assert isinstance(res0, pd.DataFrame), 'function needs to return a DataFrame'
    
    # (iii) Perform the bootstrap
    holder_bs = []
    for i in range(n_boot):
        df_bs = df.groupby(cn_gg).sample(frac=1,replace=True,random_state=i)
        df_bs = function(df_bs, **di_args).assign(idx=i)
        holder_bs.append(df_bs)
    res_bs = pd.concat(holder_bs).reset_index(drop=True)
    
    # Get the quantiles and SE
    cn_bs = list(np.setdiff1d(res_bs.columns, [cn_val, 'idx']))
    res_bs = res_bs.groupby(cn_bs)[cn_val].apply(lambda x: pd.DataFrame({'lb':x.quantile(alpha/2), 'ub':x.quantile(1-alpha/2), 'se':x.std(ddof=1)},index=[0]))
    res_bs = res_bs.reset_index().drop(columns=f'level_{len(cn_bs)}')
    # Add on a "significant" feature
    res_bs = res_bs.assign(is_sig=lambda x: np.sign(x['lb'])==np.sign(x['ub']))
    res0 = res0.merge(res_bs)
    return res0
