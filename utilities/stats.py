"""
Utility methods for statistical functions
"""

# Load modules
import numpy as np
import pandas as pd
import statsmodels.api as sm
from time import time
from typing import Callable
from sklearn.metrics import r2_score
from sklearn.base import BaseEstimator
from scipy.stats import kendalltau, spearmanr, pearsonr
from sklearn.preprocessing import StandardScaler
from statsmodels.nonparametric.kernel_regression import KernelReg
# Internal modules
from utilities.utils import str2list, cvec, merge_pairwise_dfs


def r2_score_ols(y:np.ndarray, yhat:np.ndarray, adjusted:bool=False) -> float:
    """Wrapper for getting the R-squared from a line of best fit (OLS). Can return adjusted-Rsquared with adjusted=True"""
    x = sm.add_constant(yhat)
    model = sm.OLS(y, x).fit()
    if adjusted:
        r2 = model.rsquared_adj
    else:
        r2 = model.rsquared
    return r2


def compare_r2_additive(df:pd.DataFrame, cn_gg:str or list, cn_y:str, cn_x1:str, cn_x2:str, adjusted:bool=True) -> pd.DataFrame:
    """Convenience function for calculating the additive R2 for two columns"""
    df1 = df.groupby(cn_gg).apply(lambda z: r2_score_ols(z[cn_y],z[[cn_x1, cn_x2]].values,adjusted))
    df1 = df1.reset_index().rename(columns={0:'value'}).assign(x=cn_x1)
    df12 = df.groupby(cn_gg).apply(lambda z: r2_score_ols(z[cn_y],z[cn_x1].values,adjusted))
    df12 = df12.reset_index().rename(columns={0:'value'}).assign(x=cn_x1+'_'+cn_x2)
    res = pd.concat(objs=[df1, df12], axis=0).reset_index(drop=True)
    return res


def concordance(y:np.ndarray, yhat:np.ndarray) -> float:
    """Learn to rank measure of concordance: P(s_i > s_j | y_i > y_j)"""
    yord = np.argsort(y)
    y, yhat = y[yord], yhat[yord]
    success, total = 0.0, 0.0
    for i in range(1,len(y)):
        comp_gt = yhat[i] > yhat[:i]
        comp_eq = yhat[i] == yhat[:i]
        success += sum(comp_gt) + sum(comp_eq)/2
        total += len(comp_gt)
    conc = success / total
    return conc


def somersd(y:np.ndarray, yhat:np.ndarray) -> float:
    """Transformation of concordance to the [-1,1] scale"""
    res = 2*concordance(y, yhat) - 1
    return res


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


def get_perf_msrs(df:pd.DataFrame, cn_gg:str or list, cn_y:str, cn_yhat:str, add_pearson:bool=False, add_somersd:bool=False, lm_r2:bool=False, adj_r2:bool=False, lower:None or float=None, upper:None or float=None) -> pd.DataFrame:
    """Get the key performance measures (R2, spearman, kendal, pearson, and somers-d)
    
    Parameters
    ----------
    df:             DataFrame with predicted and actual columns
    cn_gg:          Columns to groupby
    cn_y:           Name of the actual label column
    cn_yhat:        Name of the predicted label column
    add_pearson:    Whether Pearson's correlation should be added
    add_somersd:    Whether Somer's D concordance should be added
    lm_r2:          Whether the R-squared square should be based on a line of best fit
    adj_r2:         Whether the adjusted R2 score should be returned
    lower:          Whether the outputted statistic should be truncated from below
    upper:          Whether the outputted statistic should be truncated from above

    Returns
    -------
    A DataFrame with columns cn_gg, msr (e.g. 'spearman'), and value
    """
    assert isinstance(df, pd.DataFrame)
    cn_gg = str2list(cn_gg)
    if lm_r2:
        r2_fun = lambda y, x: r2_score_ols(y, x, adj_r2)
    else:
        r2_fun = r2_score
    res = df.groupby(cn_gg).apply(lambda x: pd.DataFrame({'r2':r2_fun(x[cn_y], x[cn_yhat]), 'tau':kendalltau(x[cn_y].values, x[cn_yhat].values)[0], 'rho':spearmanr(x[cn_y].values, x[cn_yhat].values)[0]},index=[0]))
    if add_pearson:
        res2 = df.groupby(cn_gg).apply(lambda x: pd.DataFrame({'pearson':pearsonr(x[cn_y].values, x[cn_yhat].values)[0]},index=[0]))
        res = pd.concat(objs=[res, res2],axis=1)
    if add_somersd:
        res3 = df.groupby(cn_gg).apply(lambda x: pd.DataFrame({'somersd':somersd(x[cn_y].values, x[cn_yhat].values)},index=[0]))
        res = pd.concat(objs=[res, res3],axis=1)
    res = res.reset_index().drop(columns=f'level_{len(cn_gg)}')
    res = res.melt(cn_gg,var_name='msr')
    if lower is not None:
        res['value'] = res['value'].clip(lower=lower)
    if upper is not None:
        res['value'] = res['value'].clip(upper=upper)
    return res


def get_perf_diff(df:pd.DataFrame, function:Callable, di_args:dict, cn_index:str or list, cn_col:str or list, cn_val:str or list) -> pd.DataFrame:
    """Convenience wrapper for calculating the difference between two factors that are outputted by some arbitrary function
    
    Parameters
    ----------
    df:                 DataFrame that neesd into first named argument in functino
    function:           Some callable function that returns a DF that can be pivoted with cn_{index,col,val}
    di_args:            Dictionary whose keys correspond to named arguments of function (e.g. function(df,**di_args))
    cn_index:           Index names
    cn_col:             Column names
    cn_val:             Value names

    Returns
    -------
    A DataFrame with columns cn_index and 'value'
    """
    # Input checks
    assert isinstance(df, pd.DataFrame)
    assert isinstance(function, Callable), 'function is not callable'
    assert all([isinstance(c, str) or isinstance(c, list) for c in str2list(cn_index) + str2list(cn_col) + str2list(cn_val)]), 'cn_{} needs be a string or a list'
    # Run function
    res = function(df, **di_args)
    # Get rowwise difference
    res = res.pivot(cn_index, cn_col, cn_val).diff(axis=1)
    # Get column with most non-missing values
    cn_max = res.notnull().sum().idxmax()
    res = res[cn_max].reset_index()
    res.rename(columns={cn_max:'value'}, inplace=True)
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


def bootstrap_function(df:pd.DataFrame, function:Callable, cn_val:str, cn_gg:str or list, n_boot:int=100, alpha:float=0.05, di_args:dict={}, verbose:bool=False) -> pd.DataFrame:
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
    stime = time()
    for i in range(n_boot):
        df_bs = df.groupby(cn_gg).sample(frac=1,replace=True,random_state=i)
        df_bs = function(df_bs, **di_args).assign(idx=i)
        holder_bs.append(df_bs)
        if verbose:
            dtime, nleft = time() - stime, n_boot - (i+1)
            rate = (i+1) / dtime
            seta = nleft / rate
            print(f'Iteration {i+1} of {n_boot} (ETA={seta/60:.2f} minutes)')
    res_bs = pd.concat(holder_bs).reset_index(drop=True)
    
    # Get the quantiles and SE
    cn_bs = list(np.setdiff1d(res_bs.columns, [cn_val, 'idx']))
    res_bs = res_bs.groupby(cn_bs)[cn_val].apply(lambda x: pd.DataFrame({'lb':x.quantile(alpha/2), 'ub':x.quantile(1-alpha/2), 'se':x.std(ddof=1)},index=[0]))
    res_bs = res_bs.reset_index().drop(columns=f'level_{len(cn_bs)}')
    # Add on a "significant" feature
    res_bs = res_bs.assign(is_sig=lambda x: np.sign(x['lb'])==np.sign(x['ub']))
    res0 = res0.merge(res_bs)
    return res0
