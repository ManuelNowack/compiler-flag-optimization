import numpy as np
import _powerset_enum as pe
from sklearn.base import BaseEstimator
from sklearn.utils.validation import  check_is_fitted
from sklearn.metrics import r2_score
from .common import SparseDSFT3Function
import cvxpy as cp
from matplotlib import pyplot as plt
import _fit
import functools

def LizEnumerator(X, Y, freq, fB, executor, recursive=True):
    if recursive:
        return executor.compute_max_correlation_recursive(Y, freq, fB)
    else:
        return executor.compute_max_correlation(Y, freq, fB)


def dualityGap(A, b, C, x, bias, dual_variable):
    m = len(b)
    primal = (1/m)*0.5*np.sum((A.dot(x) + bias - b)**2) + C*np.sum(np.abs(x)) 
    dual = (-m/2)*np.sum(dual_variable**2) - dual_variable.dot(b) #+ np.sum(bias*dual_variable)
    return primal - dual


# TODO: for the case where we have more training points we actually would like to
# solve the dual problem instead.

# we use this variant of the lasso problem because it is faster to solve
def solvePrimalQP(A, b, C=1, verbose=False, optimizer_params={}):
    m = len(b)
    p = A.shape[1]
    x = cp.Variable(p)
    bias = cp.Variable(1)
    objective = (1/m)*0.5*cp.sum_squares(A@x + bias - b) + C*cp.norm1(x)
    prob = cp.Problem(cp.Minimize(objective))
    prob.solve(**optimizer_params)# solver=cp.OSQP, polish=1, parallel=True, verbose=verbose, , eps_abs=1e-1, eps_rel=1e-1) #1e-5 is default #1e-1 is required for MRVM
    return x.value, bias.value[0], prob.value


def LPBoostPrimal(X, Y, C=1, tres=1e-8, mtres=0, steps=1000, verbose=False, est=None,
                  enumerator=pe.compute_max_correlation_recursive, force_iterations=False):
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
        freq = np.zeros(X.shape[1], dtype=np.bool)
        fB = np.zeros(X.shape[0], dtype=np.bool)
        gain = enumerator(X, z, freq, fB)
        gain = np.abs(gain)
        
        #I checked this end condition on 04.02., it should be correct
        #see writeup: Two Dual Problems to `1-Regularized Least Squares
        #the dual solution is 1/n * residual, thus, the dual feasibility constraint
        #becomes -n*C <= A^T residual <= n*C and gain is exactly the correlation
        #of the current column with the current residual
        #careful: bias term is not part of the derivation in that writeup TODO:
        #make derivation including bias term
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
            gap = dualityGap(A.T, Y, C, sol, bias, dual_variable)
            print('iteration %d, k %d, error %f, duality gap: %f'%(step, len(freqs), err, gap))

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
            #if len(errs)>0:
            #    print(np.abs(errs[-1] - err))
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

    est = SparseDSFT3Function(freqs, coefs)
    return est, errs


def lpboost_regularization_path(X, Y, k_desired, factor = 0.9, steps=100, verbose=0, patience=10):
    Y_centered = Y - Y.mean()
    executor = _fit.CorrExecutor(X, 0)
    enumerator = functools.partial(LizEnumerator, executor=executor)
    # find initial lambda
    m = X.shape[0]
    n = X.shape[1]
    freq = np.zeros(n, dtype=np.bool)
    fB = np.zeros(m, dtype=np.bool)
    corr_max = enumerator(X, Y_centered, freq, fB)
    lam_max = (1/m)*np.abs(corr_max)
    # define the range of lambdas to consider
    lam_curr = lam_max
    lams = [lam_curr]
    models = []
    n_same = 0
    k_curr = 0
    k_last = 0
    while True:
        if verbose>0:
            print('fitting ', lam_curr, '...')
        if len(models) > 0:
            est = LPBoostFast(C=lam_curr, steps=steps, est_warmstart=models[-1].est, verbose=verbose>1)
        else:
            est = LPBoostFast(C=lam_curr, steps=steps, verbose=verbose>1)
        est.fit(X, Y_centered)
        models += [est]
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
        lam_curr *= factor
        lams += [lam_curr]
    for model in models:
        model.est.coefs[0] += Y.mean()
    return models, lams


class LPBoostFast(BaseEstimator):
    def __init__(self, C=1, steps=1000, tres=1e-8, mtres=0, n_threads=0, delta=0, est=None,
                 verbose=False, force_iterations=False, est_warmstart=None):
        self.C = C
        self.steps = steps
        self.tres = tres
        self.mtres = mtres
        self.verbose = verbose
        self.force_iterations = force_iterations
        self.est_warmstart = est_warmstart
        self.n_threads = n_threads
        self.delta = delta
        self.est = est

    def fit(self, X, y):
        executor = _fit.CorrExecutor(X, self.n_threads, self.delta)
        enumerator = functools.partial(LizEnumerator, executor=executor)
        Y = y.copy()
        est, _ = LPBoostPrimal(X, Y, C=self.C, tres=self.tres, mtres=self.mtres, steps=self.steps,
                               enumerator=enumerator, verbose=self.verbose, est=self.est_warmstart,
                               force_iterations=self.force_iterations)
        self.est = est
        self.is_fitted_ = True
        return self

    def refit(self, X, y, steps=10):
        executor = _fit.CorrExecutor(X, self.n_threads, self.delta)
        enumerator = functools.partial(LizEnumerator, executor=executor)
        Y = y.copy()
        est, _ = LPBoostPrimal(X, Y, C=self.C, tres=self.tres, mtres=self.mtres, steps=steps,
                               est=self.est, enumerator=enumerator, verbose=self.verbose,
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

    
    
    
    
    
    

    
    
    
    
    
    
    
