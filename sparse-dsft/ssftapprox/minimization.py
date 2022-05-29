#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 23 17:24:12 2022

@author: chrisw
"""

from pyscipopt import Model, quicksum, multidict
import numpy as np

def polysparse_setfunction_minimization_model(freqs, coefs, C=1000.):
    """
    Implementation of Theorem 1 of https://arxiv.org/pdf/2009.10749.pdf
    """
    model = Model("Polynomial-sparse set function minimization")
    n = freqs.shape[1]
    k = freqs.shape[0]
    ones = np.ones(n)
    alpha = {}
    beta = {}
    x = {}
    for i in range(n):
        x[i] = model.addVar(vtype="B", name="x(%s)"%i)
    for i in range(k):
        beta[i] = model.addVar(vtype="B", name="beta(%s)"%i)
        alpha[i] = model.addVar(vtype="C", lb=0, ub=C, name="alpha(%s)"%i)
        freqi_dot_x = quicksum(freqs[i][j]*x[j] for j in range(n))
        model.addCons(alpha[i] >= 1 - (freqs[i].dot(ones) - freqi_dot_x))
        model.addCons(alpha[i] <= 1 - (freqs[i].dot(ones) - freqi_dot_x) + C*beta[i])
        model.addCons(alpha[i] <= C*(1 - beta[i]))
    model.setObjective(quicksum(coef * alpha[i] for i, coef in enumerate(coefs)), "minimize")
    model.data = x
    return model

def minimize_dsft3(est, C=1000.):
    """
    est: ssftapprox.common.SparseDSFT3Function
    C: parameter for the MIP, if 1000. does not work, try larger values (see https://arxiv.org/pdf/2009.10749.pdf)
    
    returns bitvector and associated function value
    """
    mip = polysparse_setfunction_minimization_model(est.freqs, est.coefs)
    mip.hideOutput()
    mip.optimize()
    bitvector = np.asarray([mip.getVal(x) for idx, x in mip.data.items()], dtype=bool)
    return bitvector, est(bitvector)[0]