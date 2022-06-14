"""
Preamble for most code and jupyter notebooks
@author: bridgetsmart
@notebook date: 6 Jun 2022
"""

import numpy as np, pandas as pd

import math, string, re, pickle, json, time, os, sys, datetime, itertools

def fit_to_markov(data):
    # need to map all to a 0,n-1 alphabet
    # then use the markov chain to get the probabilities

    # number of states
    states = [y for x in data for y in x]

    # first, get the alphabet
    alphabet = list(set(states))
    alphabet.sort()
    n = len(alphabet)
    # now, map the data to the alphabet
    data_mapped = np.array([[alphabet.index(y) for y in x] for x in data])
    states = [alphabet.index(x) for x in states]
    # get all one steps    
    one_step_trans = [(x[i], x[i+1]) for x in data_mapped for i in range(len(x)-1)]

    P = np.zeros((n, n))

    inds, counts = np.unique(one_step_trans, axis = 0, return_counts = True)
    P[inds[:, 0], inds[:, 1]] = counts

    sums = P.sum(axis = 1)
    # Avoid divide by zero error by normalizing only non-zero rows
    P[sums != 0] = P[sums != 0] / sums[sums != 0][:, None]
    
    return P, n, data_mapped

def get_ent(pi,A,n):
    # get the entropy
    ent = 0
    for i in range(n):
        for j in range(n):
            if A[i,j] ==0:
                pass
            else:
                ent += - pi[i]*A[i,j]*math.log(A[i,j])
    return ent
    
# function to get entropy estimates from LAMP fit
def get_stationary(A):
    e_vals, eig_vecs = np.linalg.eig(A.T) # gets left e vectors
    l = np.where(abs(e_vals-1) == np.min(abs(e_vals-1)))[0][0]
    # get real component
    v = eig_vecs[:,l].real
    return v/sum(v)

def ent_est(data):
    s = [y for x in data for y in x]
    #### Konty. Estimators
    #### do once by appending all
    # print('getting estimate 1...')
    # est1 = self_entropy_rate(s)

    #### do by averaging each path
    # print('getting estimate 2...')
    # est2 = np.mean([self_entropy_rate(x) for x in tqdm(data)])

    # fit markov
    print('getting markov...')
    A,n, data_mapped = fit_to_markov(data)
    pi = get_stationary(A)
    est3 = get_ent(pi,A,n)

    #### shannon estimator
    print('getting shannon from stationary dist...')
    est4 = np.sum(-pi*np.log(pi))


    ## SHANNON ESTIMATORS
    # appending all
    print('getting shannon appended...')
    seq = [y for x in data_mapped for y in x]
    est1 = get_shannon(seq)

    # averaging each path
    print('getting shannon seperate...')
    est2 = np.mean([get_shannon(x) for x in data_mapped])

    return est1, est2, est3, est4