import numpy as np
from sklearn.base import BaseEstimator
from sklearn.utils.validation import  check_is_fitted
from sklearn.metrics import r2_score
from sklearn.linear_model import ElasticNet
from .common import SparseDSFT3Function, SparseWHTFunction

import _fit
import functools
from matplotlib import pyplot as plt



def SolveElasticNet(Phi, y, elastic_net_params={'alpha':1.0, 'l1_ratio':1.0, 'max_iter':1000, 'fit_intercept':True}):
    est = ElasticNet(**elastic_net_params)
    est.fit(Phi, y)
    intercept = est.intercept_
    coefs = est.coef_
    residual = y - est.predict(Phi)
    return intercept, coefs, residual






class LowDegreeEstimator(BaseEstimator):
    def __init__(self, degree=2, tres=1e-8, enet_steps=1000, enet_alpha=1., enet_l1_ratio=1.,
                 enet_fit_intercept=True, verbose=False, n_threads=0, delta=0., recursive=False,
                 standardize=False, basis='WHT'):
        self.tres = tres
        self.verbose = verbose
        self.enet_alpha = enet_alpha
        self.enet_steps = enet_steps
        self.enet_l1_ratio = enet_l1_ratio
        self.enet_fit_intercept = enet_fit_intercept
        self.n_threads = n_threads
        self.delta = delta
        self.recursive = recursive
        self.standardize = standardize
        self.basis = basis
        self.degree = degree
        
    def _create_low_degree_support(self, n, degree=2):
        #if degree >= 0:
        #    self.support = np.zeros((1, n), dtype=np.int32)
        if degree >= 1:
            self.support = np.eye(n, dtype=np.int32)
        if degree == 2:
            pairs = []
            for i in range(n-1):
                for j in range(i, n):
                    pair = np.zeros((1, n), dtype=np.int32)
                    pair[0, i] = 1
                    pair[0, j] = 1
                    pairs += [pair]
            pairs = np.concatenate(pairs, axis=0)
            self.support = np.concatenate([self.support, pairs], axis=0)
        if degree > 2:
            raise NotImplementedError("degree higher than 2 is not implemented")
    
    def _create_features(self, X):
        if self.basis == 'WHT':
            Phi = (-1)**X.dot(self.support.T)
        if self.basis == 'DSFT3':
            Phi = X.dot(self.support.T) == self.support.sum(axis=1)
        return Phi.astype(np.float64)


    def fit(self, X, y):
        self._create_low_degree_support(X.shape[1], degree=self.degree)
        Phi = self._create_features(X)
        Y = y.copy()
        if self.standardize:
            self.Ymean = Y.mean()
            self.Ystd = Y.std()
            Y = (Y - self.Ymean)/self.Ystd
        
        elastic_net_params = {"max_iter":max(10*len(self.support), self.enet_steps), 
                              "alpha":self.enet_alpha, 
                              "l1_ratio":self.enet_l1_ratio,
                              "fit_intercept":self.enet_fit_intercept}
        est = ElasticNet(**elastic_net_params)
        est.fit(Phi, Y)
        
        mask = np.abs(est.coef_) > self.tres
        coefs = np.concatenate((np.ones(1)*est.intercept_, est.coef_[mask]), axis=0)
        freqs = np.concatenate((np.zeros((1, X.shape[1]), dtype=np.int32), self.support[mask]), axis=0)

        if self.basis == 'WHT':
            self.est = SparseWHTFunction(freqs, coefs, normalization=False)
        elif self.basis == 'DSFT3':
            self.est = SparseDSFT3Function(freqs, coefs)
        self.is_fitted_ = True
        return self

    def refit(self, X, y, steps=10):
        raise NotImplementedError('not implemented...')

    def predict(self, X):
        check_is_fitted(self, 'is_fitted_')
        Y = self.est(X)
        if self.standardize:
            Y *= self.Ystd
            Y += self.Ymean
        return Y

    def score(self, X, y):
        check_is_fitted(self, 'is_fitted_')
        return r2_score(y, self.predict(X))
    


    
    
    
    
    
    

    
    
    
    
    
    
    
