import numpy as np
import _powerset_enum as pe
from sklearn.base import BaseEstimator
from sklearn.utils.validation import  check_is_fitted
from sklearn.metrics import r2_score
from .common import SparseDSFT3Function
import _fit
import functools
from matplotlib import pyplot as plt


def LizEnumerator(X, Y, freq, fB, executor):
    return executor.compute_max_correlation(Y, freq, fB)


    
class XGBoostEstimator(BaseEstimator):
    def __init__(self, n_estimators, lr=1, row_subsampling=1, col_subsampling=1, l2=0, l1=0, max_polish=0,
                 tres=1e-6, enumerator=None, n_threads=0, est_warmstart=None, verbose=False):
        self.n_estimators = n_estimators
        self.lr = lr
        self.row_subsampling = row_subsampling
        self.col_subsampling = col_subsampling
        self.l1 = l1
        self.l2 = l2
        self.tres = tres
        self.enumerator = enumerator
        self.verbose = verbose
        self.est_warmstart = est_warmstart
        self.n_threads = n_threads
        self.max_polish = max_polish

    def fit(self, X, y):     
        #add emptyset manually
        y_mean = np.mean(y)
        ft_dict = {tuple(np.zeros(X.shape[1], dtype=bool).tolist()): y_mean}
        residual = y_mean - y # current estimate - truth
        
        change = True
        n_freqs = len(ft_dict)
        while len(ft_dict) < self.n_estimators and change:
            
            n_cols = max(1, int(X.shape[1]*self.col_subsampling))
            col_inds = np.random.permutation(X.shape[1])[:n_cols]
            n_rows = max(1, int(X.shape[0]*self.row_subsampling))
            row_inds = np.random.permutation(X.shape[0])[:n_rows]
            
            X_ = X[row_inds]
            X_ = X_[:, col_inds]
            
            residual_ = residual[row_inds]
            freq_ = np.zeros(X_.shape[1], dtype=bool) 
            fB_ = np.zeros(X_.shape[0], dtype=bool)
            executor = _fit.CorrExecutor(X_.copy(), 0)
            gain = LizEnumerator(X_, residual_, freq_, fB_, executor)
            
            freq = np.zeros(X.shape[1], dtype=bool)
            freq[col_inds] = freq_
            fB = X.astype(np.int32).dot(freq) == freq.sum() 
            gain = residual.dot(fB.astype(np.float64))
            
            if gain < -self.l1:
                alpha = self.l1
            elif gain > self.l1:
                alpha = -self.l1
            else:
                alpha = 0
            update = self.lr*(-(gain + alpha))/(self.l2 + fB.sum())
            residual += update * fB.astype(np.float64)
            ft_dict[tuple(freq.tolist())] = ft_dict.get(tuple(freq.tolist()), 0) + update
            if len(ft_dict) == n_freqs:
                residual_old = residual.copy()
                if self.max_polish > 0:
                    ft_dict, residual = self.polish(X, residual, ft_dict)
                    if self.verbose:
                        print(np.max(np.abs(residual - residual_old)))
                    if np.max(np.abs(residual - residual_old)) < self.tres:
                        change = False
                else:
                    if np.abs(update) < self.tres:
                        change = False
            n_freqs = len(ft_dict)
            if self.verbose:
                print('n_est: %d, last update: %2.6f'%(n_freqs, update))
                    
        freqs = np.asarray(list(ft_dict.keys())).astype(np.int32)
        coefs = np.asarray(list(ft_dict.values()))
        est = SparseDSFT3Function(freqs, coefs)
        self.est = est
        self.is_fitted_ = True
        return self
    
    def polish(self, X, residual, ft_dict):
        change = self.tres
        n_polish = 0
        while change >= self.tres and n_polish < self.max_polish:
            change = 0
            for key, value in ft_dict.items():
                freq = np.asarray(key)
                fB = X.astype(np.int32).dot(freq) == freq.sum()
                gain = residual.dot(fB.astype(np.float64)) 
                
                if gain < -self.l1:
                    alpha = self.l1
                elif gain > self.l1:
                    alpha = -self.l1
                else:
                    alpha = 0
                update = self.lr*(-(gain + alpha))/(self.l2 + fB.sum())
                residual += update * fB.astype(np.float64)
                ft_dict[key] += update
                change = np.maximum(change, np.abs(update))
            n_polish += 1
        return ft_dict, residual

    def refit(self, X, y, steps=10):
        raise NotImplementedError

    def predict(self, X):
        check_is_fitted(self, 'is_fitted_')
        return self.est(X)
    
    def score(self, X, y):
        check_is_fitted(self, 'is_fitted_')
        return r2_score(y, self.est(X))



    
    
    
    
    
    
    
