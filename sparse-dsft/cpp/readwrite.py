#!/usr/bin/env python
# coding: utf-8

# This script converts some of the data files to the C++ compatible format

import numpy as np
import sys

np.set_printoptions(threshold=np.inf, linewidth=np.inf)

indicators = np.load('../data/indicators18.npy')

# Small LSVM
n = 18
n_samples = 1000
for seed in range(20, 21):
    Y = np.load('../data/lsvm_seed_%d.npy'%seed)
    for idx, bidder in enumerate(Y):
        perm = np.random.permutation(2**n)
        X_train = indicators[perm[:n_samples]]
        Y_train = bidder[perm[:n_samples]]
        enfile = open("./cpp-data/small-lsvm-" + str(idx), "wb")
        enfile.write(X_train.shape[0].to_bytes(4, sys.byteorder))
        enfile.write(X_train.shape[1].to_bytes(4, sys.byteorder))
        enfile.write(X_train.astype('bool'))
        enfile.write(Y_train.astype('double'))
        enfile.close()

# Small GSVM
n = 18
n_samples = 100
for seed in range(20, 21):
    Y = np.load('../data/gsvm_seed_%d.npy'%seed)
    for idx, bidder in enumerate(Y):
        perm = np.random.permutation(2**n)
        X_train = indicators[perm[:n_samples]]
        Y_train = bidder[perm[:n_samples]]
        enfile = open("./cpp-data/small-gsvm-" + str(idx), "wb")
        enfile.write(X_train.shape[0].to_bytes(4, sys.byteorder))
        enfile.write(X_train.shape[1].to_bytes(4, sys.byteorder))
        enfile.write(X_train.astype('bool'))
        enfile.write(Y_train.astype('double'))
        enfile.close()
        
# Large MRVM
n = 98
n_samples = 500
for seed in range(20, 21):
    for idx in range(10):
        X = np.load('../data/mrvm_X_seed_%d_bidder_%d.npy'%(seed, idx))
        Y = np.load('../data/mrvm_Y_seed_%d_bidder_%d.npy'%(seed, idx))
        X_train = X[:n_samples]
        Y_train = Y[:n_samples]
        enfile = open("./cpp-data/large-mrvm-" + str(idx), "wb")
        enfile.write(X_train.shape[0].to_bytes(4, sys.byteorder))
        enfile.write(X_train.shape[1].to_bytes(4, sys.byteorder))
        enfile.write(X_train.astype('bool'))
        enfile.write(Y_train.astype('double'))
        enfile.close()
