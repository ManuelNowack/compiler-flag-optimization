import numpy as np
import _powerset_enum as pe
from sklearn.base import BaseEstimator
from sklearn.utils.validation import  check_is_fitted
from sklearn.metrics import r2_score
from .lpboost_estimators import solvePrimalQP
import cvxpy as cp
from matplotlib import pyplot as plt
import _fit
import functools


class SparseMSMeetFunction:

    def __init__(self, frequencies, coefficients):
        """
            @param frequencies: two dimensional np.array of type np.int32 or np.bool
            with one indicator vector per row
            @param coefficients: one dimensional np.array of corresponding Fourier
            coeffients
        """
        self.freqs = frequencies
        self.coefs = coefficients
        self.call_counter = 0

    def __call__(self, indicators, count_flag=True):
        """
            @param indicators: two dimensional np.array of type np.int32 or np.bool
            with one indicator vector per row
            @param count_flag: a flag indicating whether to count set function evaluations
            @returns: a np.array of set function evaluations
        """
        ind = indicators
        freqs = self.freqs
        coefs = self.coefs


        if len(ind.shape) < 2:
            ind = ind[np.newaxis, :]

        FREQS = np.tile(freqs, [1, len(ind)]).reshape(len(freqs), len(ind), -1)
        INDS = np.tile(ind, [len(freqs), 1]).reshape(len(freqs), len(ind), -1)
        res = np.all(FREQS <= INDS, axis=-1)
        return res.astype(np.float64).T.dot(coefs)


def MSLPBoostPrimal(X, Y, C=1, tres=1e-8, mtres=0, steps=1000, verbose=False, est=None,
                  enumerator=pe.ms_compute_max_correlation, force_iterations=False):
    mfact = 1+mtres
    n = len(X)
    sol = None
    bias = None
    if est is None:
        z = Y.copy()
        A = np.empty((0, len(Y)))
        freqs = np.empty((0, X.shape[1]), dtype=np.int32)
        freqs_set = set()
    else:
        freqs = est.freqs
        coefs = est.coefs
        A = [(X.dot(freq) == freq.sum()).astype(np.float64) for freq in freqs]
        A = np.asarray(A)
        z = est(X) - Y
        freqs_set = set([tuple(freq.tolist()) for freq in freqs])

        # already refit the CVX opt one time
        try:
            sol, bias, err = solvePrimalQP(A.T, Y, C=C)
            z = A.T.dot(sol) + bias - Y
        except:
            print('solving the lasso problem for the new data and the old support failed')


    errs = []
    for step in range(steps):
        freq = np.zeros(X.shape[1], dtype=np.int32)
        fB = np.zeros(X.shape[0], dtype=np.bool)
        gain = enumerator(X, z, freq, fB)
        gain = np.abs(gain)

        if verbose:
            print('iteration %d: feasible %2.4f > %2.4f?:'%(step, gain, C*n*mfact), gain > C*n*mfact)
            if gain <= C*n*mfact:
                print('constraint endcondition fires')
        
        if not force_iterations and gain <= C*n*mfact:
            if verbose:
                print('new column violates the constraints of the optimization problem')
            break
 
            
        new_freq = tuple(freq.astype(np.int32).tolist())
        if new_freq not in freqs_set:
            freqs = np.concatenate([freqs, freq[np.newaxis, :]], axis=0)
            A = np.concatenate([A, fB[np.newaxis, :].astype(np.float64)], axis=0)

        try:
            sol_old = sol
            bias_old = bias
            sol, bias, err = solvePrimalQP(A.T, Y, C=C)
            freqs_set.add(new_freq)
        except:
            if new_freq not in freqs_set:
                freqs = freqs[:-1]
                A = A[:-1]
                sol = sol_old
                bias = bias_old
            print('OPTIMIZATION FAILED')
            break
        dual_variable = (A.T.dot(sol) + bias - Y)/n
        if verbose:
            print('iteration %d, k %d, error %f'%(step, len(freqs), err))

        if sol_old is not None:
            if len(sol_old) == len(sol):
                sol_diff = np.sqrt(np.sum((sol_old - sol)**2)) #+ (bias - bias_old)**2)
                fit_diff = np.linalg.norm(A.T.dot(sol) - A.T.dot(sol_old))
            else:
                sol_diff = np.sqrt(np.sum((sol_old - sol[:-1])**2) + sol[-1]**2 )
                fit_diff = np.linalg.norm(A.T.dot(sol) - A[:-1].T.dot(sol_old))
            relative_solution_norm_change = sol_diff / np.sqrt(np.sum(sol**2))
            relative_fit_norm_change = fit_diff / np.linalg.norm(A.T.dot(sol))
        if not force_iterations and sol_old is not None:
            if relative_solution_norm_change < tres:
                if verbose:
                    print('relative solution norm changed less than %f'%tres)
                    print(relative_solution_norm_change, relative_fit_norm_change)
                    plt.plot(np.abs(A.dot(A.T.dot(sol)) - A.dot(Y - bias)))
                    plt.hlines(C * n, 0, len(sol))
                    plt.show()
                    print((np.abs(A.dot(A.T.dot(sol)) - A.dot(Y - bias)) - C * n < tres).sum())
                break
            if relative_fit_norm_change < tres:
                if verbose:
                    print('relative fit norm changed less than %f' % tres)
                    print(relative_solution_norm_change, relative_fit_norm_change)
                    plt.plot(np.abs(A.dot(A.T.dot(sol)) - A.dot(Y - bias)))
                    plt.hlines(C * n, 0, len(sol))
                    plt.show()
                    print((np.abs(A.dot(A.T.dot(sol)) - A.dot(Y - bias)) - C * n < tres).sum())
                break
            if len(sol) >= n-1:
                if verbose:
                    print('n = %d columns found...'%n)
                    plt.plot(np.abs(A.dot(A.T.dot(sol)) - A.dot(Y - bias)))
                    plt.hlines(C*n, 0, len(sol))
                    plt.show()
                    print((np.abs(A.dot(A.T.dot(sol)) - A.dot(Y - bias)) - C*n < tres).sum())
                break


        z = A.T.dot(sol) + bias - Y
        A = A[np.abs(sol) > 1e-6]
        freqs = freqs[np.abs(sol) > 1e-6]
        sol = sol[np.abs(sol) > 1e-6]

        if verbose:
            if sol_old is not None and np.abs(np.linalg.norm(sol_old) - np.linalg.norm(sol)) + np.abs(bias - bias_old) < tres:
                print('solution norm endcondition fires')

        errs += [err]
        if verbose:
            print('---------------------')
    if sol is not None:
        freqs = np.asarray(freqs)
        freqs = freqs[np.abs(sol) > 1e-6]
        coefs = sol[np.abs(sol) > 1e-6]    
        if bias is not None:
            freqs = np.concatenate([np.zeros((1, X.shape[1]), dtype=np.int32), freqs], axis=0)
            coefs = np.concatenate([np.ones(1)*bias, coefs])
    elif est is not None:
        return est, None
    else:
        freqs = np.zeros((1, X.shape[1]), dtype=np.int32)
        coefs = np.zeros(1)

    est = SparseMSMeetFunction(freqs, coefs)
    return est, errs


class MSLPBoostEstimator(BaseEstimator):
    def __init__(self, multiplicities, C=1, steps=1000, tres=1e-8, mtres=0, enumerator=pe.ms_compute_max_correlation,
                 verbose=False, force_iterations=False):
        self.multiplicities = multiplicities
        self.C = C
        self.steps = steps
        self.tres = tres
        self.mtres = mtres
        self.enumerator = functools.partial(enumerator, M=self.multiplicities)
        self.verbose=verbose
        self.force_iterations = force_iterations
        
    def fit(self, X, y):
        Y = y.copy()
        est, _  = MSLPBoostPrimal(X, Y, C=self.C, tres=self.tres, mtres=self.mtres, steps=self.steps,
                                enumerator=self.enumerator, verbose=self.verbose, 
                                force_iterations=self.force_iterations)
        self.est = est
        self.is_fitted_ = True
        return self
    
    def refit(self, X, y, steps=10):
        Y = y.copy()            
        est, _  = MSLPBoostPrimal(X, Y, C=self.C, tres=self.tres, mtres=self.mtres, steps=steps, est=self.est,
                                enumerator=self.enumerator, verbose=self.verbose,
                                force_iterations=self.force_iterations)
        self.est = est
        self.is_fitted_ = True
        return self

    def predict(self, X):
        check_is_fitted(self, 'is_fitted_')
        Y = self.est(X)
        return Y
    
    def score(self, X, y):
        check_is_fitted(self, 'is_fitted_')
        return r2_score(y, self.predict(X))


    
    
    
    
    
    

    
    
    
    
    
    
    
