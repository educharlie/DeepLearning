
# coding=utf-8

import time
import numpy as np  

def L1(yhat, y):
    
    return np.sum(abs(yhat - y))

def L2(yhat, y):
    
    return np.sum(np.dot(yhat - y, yhat - y))

yhat = np.array([.9, 0.2, 0.1, .4, .9])
y = np.array([1, 0, 0, 1, 1])
print("L1 = " + str(L1(yhat,y)))

yhat = np.array([.9, 0.2, 0.1, .4, .9])
y = np.array([1, 0, 0, 1, 1])
print("L2 = " + str(L2(yhat,y)))