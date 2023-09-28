import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from time import time
import pandas as pd
from utils import *

# Kar 2013  and OGD 2021 algorithms  with RSx sampling or FIFO sampling 
def LimBuffer(X,Y,X_test,Y_test,options,kar=False):
  np.random.seed(0)
  eta = options.eta
  lmd = options.lmd
  s = options.s
  n,d = X.shape
  w = np.random.randn(d,1)*0
  st = time()
  Tims_Y = [0]
  AUCs_YIMING = [0.5]
  loss = []
  x_p = X[np.where(Y==1)[0]][0] ; x_p = x_p.toarray()
  x_n = X[np.where(Y==-1)[0]][0]  ; x_n = x_n.toarray()
  Bp = x_p.T
  Bn = x_n.T
  Hp = Bp
  Hn = Bn
  tp = 1
  tn = 1
  for k in range(options.epoch):
    i = k
    x = X[[i],:] ; x = x.toarray()
    x = x.T
    if Y[i] == 1:
      Dt = x - Bn
      Bp = RS_buffer(Bp,x,s,tp,kar)
      tp += 1
    else:
      Dt = Bp - x
      Bn = RS_buffer(Bn,x,s,tn,kar)
      tn += 1
    ls , g = grad(w,Dt)
    w = w - eta *  g
    w = np.sign(w) * np.maximum(np.abs(w) - lmd * eta , 0 )
    loss.append(ls)
    if k%50 == 0:
      Tims_Y.append(time()-st)
      auc = roc_auc_score(Y_test, 1 / (  1 +  np.clip( np.exp(-X_test.dot(w)) ,0.001,10  )    ))
      AUCs_YIMING.append(auc)
  return np.array(Tims_Y),np.array(AUCs_YIMING),loss

# No clustering s-Buffer update policy
def RS_buffer(B,x,s,t,kar=False):
    if kar == False:
        return x # Limited Buffer with P=1
    if t < s:
        B = np.c_[B,x]
    else:
        for i in range(B.shape[1]):
            if np.random.binomial(1,1/t) == 1: 
                B[:,[i]] = x
    return B