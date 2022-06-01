#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 23 17:24:12 2022

@author: chrisw
"""

from pyscipopt import Model, quicksum, multidict
import numpy as np

def polysparse_setfunction_minimization_model(freqs, coefs, C=1000., cardinality_constraint = None):
    """
    Implementation of Theorem 1 of https://arxiv.org/pdf/2009.10749.pdf
    
    cardinality_constraint: function that evaluates to true if the cardinality constraint is met
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
        freqi_dot_x = quicksum(freqs[i, j]*x[j] for j in range(n))
        model.addCons(alpha[i] >= 1 - (freqs[i].dot(ones) - freqi_dot_x))
        model.addCons(alpha[i] <= 1 - (freqs[i].dot(ones) - freqi_dot_x) + C*beta[i])
        model.addCons(alpha[i] <= C*(1 - beta[i]))
    if cardinality_constraint is not None:
        model.addCons(cardinality_constraint(quicksum(x[i] for i in range(n))))
    model.setObjective(quicksum(coef * alpha[i] for i, coef in enumerate(coefs)), "minimize")
    model.data = x
    return model

def minimize_dsft3(est, C=1000., cardinality_constraint = None):
    """
    est: ssftapprox.common.SparseDSFT3Function
    C: parameter for the MIP, if 1000. does not work, try larger values (see https://arxiv.org/pdf/2009.10749.pdf)
    
    returns bitvector and associated function value
    """
    mip = polysparse_setfunction_minimization_model(est.freqs, est.coefs, C=C, cardinality_constraint=cardinality_constraint)
    mip.hideOutput()
    mip.optimize()
    bitvector = np.asarray([mip.getVal(x) for idx, x in mip.data.items()], dtype=bool)
    return bitvector, est(bitvector)[0]

def dsft4sparse_setfunction_minimization_model(freqs, coefs, C=1000., cardinality_constraint = None):
    """
    Implementation of Theorem 1 of https://arxiv.org/pdf/2009.10749.pdf
    
    cardinality_constraint: function that evaluates to true if the cardinality constraint is met
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
        freqi_dot_x = quicksum(freqs[i, j]*x[j] for j in range(n))
        model.addCons(alpha[i] >= 1 - freqi_dot_x)
        model.addCons(alpha[i] <= 1 - freqi_dot_x + C*beta[i])
        model.addCons(alpha[i] <= C*(1 - beta[i]))
    if cardinality_constraint is not None:
        model.addCons(cardinality_constraint(quicksum(x[i] for i in range(n))))
    model.setObjective(quicksum(coef * alpha[i] for i, coef in enumerate(coefs)), "minimize")
    model.data = x
    return model

def minimize_dsft4(est, C=1000., cardinality_constraint = None):
    """
    est: ssftapprox.common.SparseDSFT4Function
    C: parameter for the MIP, if 1000. does not work, try larger values (see https://arxiv.org/pdf/2009.10749.pdf)
    
    returns bitvector and associated function value
    """
    mip = dsft4sparse_setfunction_minimization_model(est.freqs, est.coefs, C=C, cardinality_constraint=cardinality_constraint)
    mip.hideOutput()
    mip.optimize()
    bitvector = np.asarray([mip.getVal(x) for idx, x in mip.data.items()], dtype=bool)
    return bitvector, est(bitvector)[0]

def whtsparse_setfunction_minimization_model(freqs, coefs, cardinality_constraint = None):
    """
    Implementation of Theorem 1 of https://arxiv.org/pdf/2009.10749.pdf
    
    cardinality_constraint: function that evaluates to true if the cardinality constraint is met
    """
    model = Model("Polynomial-sparse set function minimization")
    n = freqs.shape[1]
    k = freqs.shape[0]
    ones = np.ones(n)
    beta = {}
    gamma = {}
    x = {}
    for i in range(n):
        x[i] = model.addVar(vtype="B", name="x(%s)"%i)
    for i in range(k):
        beta[i] = model.addVar(vtype="B", name="beta(%s)"%i)
        gamma[i] = model.addVar(vtype="I", lb=0, ub=None, name="gamma(%s)"%i)
        freqi_dot_x = quicksum(freqs[i, j]*x[j] for j in range(n))
        model.addCons(beta[i] == freqi_dot_x - 2*gamma[i])
    if cardinality_constraint is not None:
        model.addCons(cardinality_constraint(quicksum(x[i] for i in range(n))))
    model.setObjective(quicksum(coef * (-2*beta[i] + 1) for i, coef in enumerate(coefs)), "minimize")
    model.data = x
    return model

def minimize_wht(est, cardinality_constraint = None):
    """
    est: ssftapprox.common.SparseWHTFunction
    
    returns bitvector and associated function value
    """
    mip = whtsparse_setfunction_minimization_model(est.freqs, est.coefs, cardinality_constraint=cardinality_constraint)
    mip.hideOutput()
    mip.optimize()
    bitvector = np.asarray([mip.getVal(x) for idx, x in mip.data.items()], dtype=bool)
    return bitvector, est(bitvector)[0]