import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from time import time
import pandas as pd
from scipy.sparse import csr_matrix
from utils import *



def Mixture(X,Y,X_test,Y_test,options):
  eta = options.eta
  lmd = options.lmd
  s = options.s
  n,d = X.shape
  np.random.seed(0)
  w = np.random.randn(d,1)*0.0
  gp = np.zeros((d,1))
  st = time()
  loss = []

  Tims_Y = [0]
  AUCs_YIMING = [0]
  x_p = X[np.where(Y==1)[0]][0] ; x_p = x_p.toarray()
  x_n = X[np.where(Y==-1)[0]][0]; x_n = x_n.toarray()
  Bp = x_p.T
  Bn = x_n.T
  BCp = Bp
  BCn = Bn
  Hp = Bp
  Hn = Bn
  numP = np.ones(s)
  numN = np.ones(s)
  tp = 1
  tn = 1
  for k in range(options.epoch):
    i = k
    x = X[[i],:] ; x = x.toarray()
    x = x.T
    if Y[i] == 1:
      Dt = x - Bn
      BCp , Bp, numP = SF_buffer(BCp, Bp,x,s,numP)
      ls , g = grad(w,Dt,numP,tp)
      tp +=1
    else:
      Dt = Bp - x
      BCn , Bn,numN = SF_buffer(BCn,Bn,x,s,numN)
      ls , g = grad(w,Dt,numN,tn)
      tn+=1
    ww = w - eta * g
    w = np.sign(ww) * np.maximum(np.abs(ww) - lmd * eta , 0 )
    loss.append(ls)
    if k%50 == 0:
      Tims_Y.append(time()-st)
      auc = roc_auc_score(Y_test, 1 / (  1 +  np.clip( np.exp(-X_test.dot(w)) ,0.001,10  )    ))
      AUCs_YIMING.append(auc)
  return np.array(Tims_Y),np.array(AUCs_YIMING),loss

def gridsearchM(X_train,Y_train,s):
  optg = options()
  optg.s = s
  b = 0
  n,d = X_train.shape
  split = n - int(n/4)
  for i in range(-10,-3):
    eta = 2**(i)
    for j in range(-9,-6):
      lmd = 10**j
      optg.eta = eta
      optg.lmd = lmd
      if X_train[:split,:].shape[0] <= 4000:
        optg.epoch = X_train[:split,:].shape[0]
      else:
        optg.epoch = 4000
      T,A,w = Mixture(X_train[:split,:],Y_train[:split,:],X_train[split:,:],Y_train[split:,:],optg)
      if (A[-1]) > b: 
        b = A[-1]
        eta_best = eta
        lmd_best = lmd
  return eta_best,lmd_best

