import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from time import time
import pandas as pd



# Hinge loss
def grad(w,Z,num=np.ones(10000),t=1):
    err = 2 - w.T@Z
    err = np.maximum(0,err)
    sumg = 0
    for i in range(Z.shape[1]):# *
       sumg+= -  num[i]/t * (1-w.T@Z[:,i]) * Z[:,i]
    dsqrr = sumg/Z.shape[1]
    return np.mean(err),dsqrr.reshape(-1,1)

class options:
  eta = None
  lmd = None
  epoch = None

# estimate epsion 
def estimate_epsilon(X):
    est = 0
    n = X.shape[0]
    for _ in range(1000):
        i = np.random.randint(n)
        j = np.random.randint(n)
        est += np.linalg.norm(X[i].toarray()-X[j].toarray())**2
    
    return est/1000

# # Clusters Based Buffer update
def SF_buffer(C,B,x,k,eps,num=np.ones(100)):
    diff = np.sum((C-x)**2,0)
    if np.min(diff) >= eps and C.shape[1] < k:
      C = np.c_[C,x]
      B = np.c_[B,x]
    else: 
      J = np.argmin(diff)
      C[:,[J]] = C[:,[J]] + 0.05 * (C[:,[J]] - x) 
      B[:,[J]] = x
      num[J] +=1
    return C,B,num

def smooth(Var,mu=0.95):
  g_history = np.array(Var)
  g_avg = np.zeros_like(g_history)
  # Average
  for i in range(0,g_history.size):
    g_avg[i] = (mu*g_avg[i-1] + (1-mu) * g_history[i])   # g_history[i] + mu * (g_avg[i-1] - g_history[i])
  # Remove Bias
  for i in range(0,g_history.size):
    g_avg[i] *= 1/(1-mu**(i+1))
  return g_avg

def moving_average(data, window_size):
    return np.convolve(data, np.ones(window_size), 'valid') / window_size