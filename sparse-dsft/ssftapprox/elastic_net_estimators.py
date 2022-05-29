#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 23 17:37:47 2022

@author: chrisw
"""

import numpy as np
import _powerset_enum as pe
from sklearn.base import BaseEstimator
from sklearn.utils.validation import  check_is_fitted
from sklearn.metrics import r2_score
from sklearn.linear_model import ElasticNet
from .common import SparseDSFT3Function

import _fit
import functools
from matplotlib import pyplot as plt

def LizEnumerator(X, Y, freq, fB, executor, recursive=False):
    if recursive:
        return executor.compute_max_correlation_recursive(Y, freq, fB)
    else:
        return executor.compute_max_correlation(Y, freq, fB)


def SolveElasticNet(Phi, y, elastic_net_params={'alpha':1.0, 'l1_ratio':1.0, 'max_iter':10000, 'fit_intercept':True}):
    est = ElasticNet(**elastic_net_params)
    est.fit(Phi, y)
    intercept = est.intercept_
    coefs = est.coef_
    residual = y - est.predict(Phi)
    return intercept, coefs, residual


def SeriesOfElasticNet(X, Y, tres=1e-8, steps=1000, verbose=False,
                  enumerator=pe.compute_max_correlation,
                  elastic_net_params={'alpha':1.0, 'l1_ratio':1.0, 'max_iter':1000, 'fit_intercept':True},
                  initial_support=None):

    if initial_support is None:
        residual = Y.copy()
        PhiT = np.ones((0, len(Y)))
        freqs = np.zeros((0, X.shape[1]), dtype=np.int32)
        freqs_set = set()
    else:
        freqs = initial_support.astype(np.int32)
        freqs_set = set(tuple(freq.tolist()) for freq in initial_support)
        Phi = X.dot(freqs.T) == freqs.sum(axis=1)[np.newaxis, :]
        PhiT = 2*Phi.T.astype(np.float64)-1
        elastic_net_params['max_iter'] = 10*len(freqs)
        intercept, coefs, residual = SolveElasticNet(PhiT.T, Y, elastic_net_params)
        #plt.matshow(Phi)
        #plt.show()
    errs = []
    
    intercept = None
    coefs = None
    
    for step in range(steps):
        freq = np.zeros(X.shape[1], dtype=bool)
        fB = np.zeros(X.shape[0], dtype=bool)
        gain = enumerator(X, residual, freq, fB)
        gain = np.abs(gain)
        
        new_freq = tuple(freq.astype(np.int32).tolist())
        if new_freq not in freqs_set:
            freqs = np.concatenate([freqs, freq[np.newaxis, :]], axis=0)
            PhiT = np.concatenate([PhiT, 2*fB[np.newaxis, :].astype(np.float64) - 1], axis=0)

        
        intercept_old = intercept
        coefs_old = coefs
        
        #print(PhiT.shape)
        elastic_net_params['max_iter'] = 10*len(freqs)
        intercept, coefs, residual = SolveElasticNet(PhiT.T, Y, elastic_net_params)
        freqs_set.add(new_freq)
        
        #print(coefs)
        
        if coefs_old is not None:
            if len(coefs_old) == len(coefs):
                sol_diff = np.sqrt(np.sum((coefs_old - coefs)**2)) 
            else:
                sol_diff = np.sqrt(np.sum((coefs_old - coefs[:-1])**2) + coefs[-1]**2)
            sol_diff += (intercept - intercept_old)**2
            relative_solution_norm_change = sol_diff / (tres + np.sqrt(np.sum(coefs**2) + intercept**2))

            if relative_solution_norm_change < tres:
                break
        
        if len(PhiT) > 100:
            mask = np.abs(coefs) > tres
            PhiT = PhiT[mask]
            freqs = freqs[mask]
            coefs = coefs[mask]

        errs += [np.linalg.norm(residual)]
        
    if coefs is not None:
        mask = np.abs(coefs) > tres
        freqs = freqs[mask]
        coefs = coefs[mask] 
        
        freqs = np.concatenate([freqs, np.zeros((1, X.shape[1]), dtype=np.int32)], axis=0)
        coefs = np.concatenate([2*coefs, np.ones(1)*(intercept - coefs.sum())])
    else:
        freqs = np.zeros((1, X.shape[1]), dtype=np.int32)
        coefs = np.zeros(1)

    est = SparseDSFT3Function(freqs, coefs)
    return est, errs



class ElasticNetEstimator(BaseEstimator):
    def __init__(self, steps=1000, tres=1e-8, enet_steps=1000, enet_alpha=1., enet_l1_ratio=1.,
                 enet_fit_intercept=True, verbose=False, n_threads=0, delta=0., recursive=False,
                 standardize=False, initial_support=None, create_low_degree_support=False, degree=2):
        self.steps = steps
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
        self.initial_support = initial_support
        self.create_low_degree_support = create_low_degree_support
        self.degree = degree
        
    def _create_low_degree_support(self, n, degree=2):
        #if degree >= 0:
        #    self.initial_support = np.zeros((1, n), dtype=np.int32)
        if degree >= 1:
            self.initial_support = np.eye(n, dtype=np.int32)
        if degree == 2:
            pairs = []
            for i in range(n-1):
                for j in range(i, n):
                    pair = np.zeros((1, n), dtype=np.int32)
                    pair[0, i] = 1
                    pair[0, j] = 1
                    pairs += [pair]
            pairs = np.concatenate(pairs, axis=0)
            self.initial_support = np.concatenate([self.initial_support, pairs], axis=0)
        if degree > 2:
            raise NotImplementedError("degree higher than 2 is not implemented")


    def fit(self, X, y):
        if self.create_low_degree_support:
            self._create_low_degree_support(X.shape[1], degree=self.degree)
        elastic_net_params = {"max_iter":self.enet_steps, 
                              "alpha":self.enet_alpha, 
                              "l1_ratio":self.enet_l1_ratio,
                              "fit_intercept":self.enet_fit_intercept}
        executor = _fit.CorrExecutor(X, self.n_threads, self.delta)
        enumerator = functools.partial(LizEnumerator, executor=executor, recursive=self.recursive)
        Y = y.copy()
        if self.standardize:
            self.Ymean = Y.mean()
            self.Ystd = Y.std()
            Y = (Y - self.Ymean)/self.Ystd
        est, _ = SeriesOfElasticNet(X, Y, tres=self.tres, steps=self.steps,
                               enumerator=enumerator, verbose=self.verbose, 
                               elastic_net_params=elastic_net_params, 
                               initial_support=self.initial_support)

        self.est = est
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
    
