import numpy as np
import _powerset_enum as pe
from sklearn.base import BaseEstimator
from sklearn.utils.validation import  check_is_fitted
from sklearn.metrics import r2_score
from .common import SparseDSFT3Function
import _fit
import functools


def LizEnumerator(X, Y, freq, fB, executor):
    return executor.compute_max_correlation(Y, freq, fB)


def prox(w0, C):
    return max(1 - C/abs(w0), 0)*w0


def coordinate_descent_fixed(X, Y, params, feature_vecs, max_rounds=100, C=1, tres=1e-6, center=True, verbose=False):
    m = len(Y)
    residual = Y.copy()
    for polish in range(max_rounds):
        updates = []
        for key, Aj in list(feature_vecs.items()):
            if key == tuple([0]*X.shape[1]):
                continue
            if center:
                Aj = Aj - Aj.mean()
            Hjj = 1/m * np.dot(Aj, Aj) #hessian
            #print(np.asarray(key, dtype=np.int32))
            gradj = 1/m * np.dot(Aj, -residual) #Xw - y = - (y - Xw)
            wj = params.get(key, 0)
            wj_opt = prox(wj - gradj/Hjj, C/Hjj)
            params[key] = wj_opt # wj_opt = wj + (wj_opt - wj)
            updates += [wj_opt - wj]
            if np.abs(wj_opt) < tres:
                if verbose:
                    print('removing', key)
                params.pop(key, None)
                feature_vecs.pop(key, None)

            residual = residual - (wj_opt - wj)*Aj
        if len(updates) == 0 or np.max(np.abs(updates)) < tres:
            if verbose:
                print('converged after %d polish steps'%(polish+1))
            break
    
    return params, feature_vecs, residual
    


def coordinate_descent(X, Y, steps, C=1, tres=1e-6, patience=10, n_polish=10, polish_per=1, est=None, center=True, verbose=False, enumerator=pe.compute_max_correlation, correct_residual=False):
    """correct_residual is a boolean flag indicating whether the residual needs to be corrected w.r.t. the bias """
    m = len(Y)
    # if est is not none we do a hot start
    if est is None:
        params = {}
        feature_vecs = {}
        residual = Y.copy()
    else:
        params = {tuple(freq.astype(np.int32).tolist()): coef for freq, coef in zip(est.freqs, est.coefs)}
        feature_vecs = {tuple(freq.astype(np.int32).tolist()): (X.dot(freq) == freq.sum()).astype(np.float64) for freq in est.freqs}
        if np.abs(Y.mean()) < tres and correct_residual:
            params.pop(tuple([0]*X.shape[1]), None)
            feature_vecs.pop(tuple([0]*X.shape[1]), None)
            if verbose:
                print('Y is centered')
            residual = Y - est(X) + est(np.zeros(X.shape[1], np.bool))
        else:
            residual = Y - est(X) 


        params, feature_vecs, residual = coordinate_descent_fixed(X, residual, params, feature_vecs, C=C, max_rounds=n_polish, tres=tres, center=center, verbose=verbose)
            
        
    no_improvement_since = 0
    for step in range(steps):
        freq = np.zeros(X.shape[1], dtype=bool)
        fB = np.zeros(X.shape[0], dtype=bool)
        gain = enumerator(X, residual, freq, fB)
        key = tuple(freq.astype(np.int32).tolist())
        
        # update weight
        feature_vecs[key] = fB
        Aj = fB.astype(np.float64)
        if center:
            Aj = Aj - Aj.mean()
        
        Hjj = 1/m * np.dot(Aj, Aj) #hessian
        gradj = 1/m * np.dot(Aj, -residual) #Xw - y = - (y - Xw)
        wj = params.get(key, 0)
        wj_opt = prox(wj - gradj/Hjj, C/Hjj)
        params[key] = wj_opt # wj_opt = wj + (wj_opt - wj)
        
        residual = residual - (wj_opt - wj)*Aj
        
        #stop the expensive looking for new frequencies when no new keys are added
        if abs(wj - wj_opt) < tres:
            no_improvement_since += 1
            params, feature_vecs, residual = coordinate_descent_fixed(X, residual, params, feature_vecs,C=C,  max_rounds=n_polish, tres=tres, center=center, verbose=verbose)
            if no_improvement_since > patience:
                break
        else:
            no_improvement_since = 0
        
        if polish_per is not None:
            if step % polish_per == 0:
                params, feature_vecs, residual = coordinate_descent_fixed(X, residual, params, feature_vecs,C=C,  max_rounds=n_polish, tres=tres, center=center, verbose=verbose)

    
    if center:
        correction = 0
        for key, fB in feature_vecs.items():
            correction -= params[key]*fB.mean()
        key = tuple([0]*X.shape[1])
        w0 = params.get(key, 0)
        params[key] = correction + w0
    
    
    if len(params) == 0:
        params[tuple(X.shape[1]*[0])] = 0
    if verbose:
        print('%d steps were required done...'%step)
    freqs = np.asarray(list(params.keys()))
    coefs = np.asarray(list(params.values()))

    est = SparseDSFT3Function(freqs, coefs)
    return est


def coordinate_descent_regularization_path(X, Y, k_desired, factor = 0.9, patience=10, steps=1000, tres=1e-6, n_polish=10, 
                                           polish_per=None, verbose = 0):
    ymean = Y.mean()
    Y_centered = Y - ymean
    executor = _fit.CorrExecutor(X, 0)
    enumerator = functools.partial(LizEnumerator, executor=executor)
    # find initial lambda
    m = X.shape[0]
    n = X.shape[1]
    freq = np.zeros(n, dtype=np.bool)
    fB = np.zeros(m, dtype=np.bool)
    corr_max = enumerator(X, Y_centered, freq, fB)
    lam_max = (1/m)*np.abs(corr_max)
    lam = lam_max
    lams = []
    models = []
    n_same = 0
    k_curr = 0
    k_last = 0
    while True:
        # solve the problem using coordinate descent
        if verbose>0:
            print('solving for lamda', lam)
        if len(models) == 0:
            est_prev = None
        else:
            est_prev = models[-1].est
        est = LassoEstimator(C=lam, steps=steps, patience=patience, tres=tres, n_polish=n_polish,
                             polish_per=polish_per, est_warmstart=est_prev, verbose=verbose>1)
        est.fit(X, Y_centered)
        
        models += [est]
        lams += [lam]
        lam *= factor
        
        k_last = k_curr
        k_curr = len(est.est.coefs)
        if k_last == k_curr:
            n_same += 1
        else:
            n_same = 0
        if verbose>0:
            print('#coefficients fit:', k_curr)
        if k_curr > k_desired:
            break
        if n_same >= patience:
            break

        
    for model in models:
        for i in range(model.est.coefs.size):
            if np.count_nonzero(model.est.freqs[i]) == 0:
                model.est.coefs[i] += ymean
                break
    return models, lams

    
class LassoEstimator(BaseEstimator):
    """"maybe my lasso can be improved by just looking at https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html"""
    def __init__(self, C=1, steps=10000, patience=3, tres=1e-4, n_polish=10, polish_per=None, enumerator=None, est_warmstart=None, verbose=False):
        """ if enumerator is None, we use the fast implementation """
        self.C = C
        self.steps = steps
        self.tres = tres
        self.n_polish = n_polish
        self.patience = patience
        self.enumerator = enumerator
        self.verbose=verbose
        self.est_warmstart = est_warmstart
        self.polish_per=polish_per

    def fit(self, X, y):
        if self.enumerator is None:
            executor = _fit.CorrExecutor(X, 0)
            enumerator = functools.partial(LizEnumerator, executor=executor)
        else:
            enumerator = self.enumerator

        y_centered = y - y.mean()
        est = coordinate_descent(X, y_centered, self.steps, self.C, self.tres, self.patience, self.n_polish, 
                                 polish_per = self.polish_per,
                                 enumerator=enumerator, est=self.est_warmstart, verbose=self.verbose)

        est.coefs[-1] += y.mean()  # last coef is the one of emptyset (see coordinate dcescent function)
        self.est = est
        self.is_fitted_ = True
        # `fit` should always return `self`
        return self

    def refit(self, X, y, steps=10):
        if self.enumerator is None:
            executor = _fit.CorrExecutor(X, 0)
            enumerator = functools.partial(LizEnumerator, executor=executor)
        else:
            enumerator = self.enumerator
            
        y_centered = y - y.mean()
        est = coordinate_descent(X, y_centered, steps, self.C, self.tres, self.patience, self.n_polish,
                                 polish_per = self.polish_per,
                                 enumerator=enumerator, est=self.est, verbose=self.verbose)
        for i in range(est.coefs.size):
            if np.count_nonzero(est.freqs[i]) == 0:
                est.coefs[i] += y.mean()
                break
        self.est = est
        self.is_fitted_ = True
        return self

    def predict(self, X):
        check_is_fitted(self, 'is_fitted_')
        return self.est(X)
    
    def score(self, X, y):
        check_is_fitted(self, 'is_fitted_')
        return r2_score(y, self.est(X))



    
    
    
    
    
    
    
