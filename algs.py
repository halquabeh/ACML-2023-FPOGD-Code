# Online gradioent descent with linear models, and stratified sampling (online clustering), ACML23
#Initialization 
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

from joblib import Memory
from sklearn.datasets import load_svmlight_file
mem = Memory("./mycache")
from sklearn.preprocessing import normalize
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

@mem.cache
def get_data(filePath):
    data = load_svmlight_file(filePath)
    X = data[0]; Y = data[1]
    X = 2*normalize(X, norm='l2', axis=1) # normalize such that each row will have \|\cdot\|_2 = 1
    Y = Y.reshape(-1,1)
    # X = X.toarray()
    # split into train-test sets
    # X, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=123)
    return X,Y

# square loss 
def grad(w,Z,num=np.ones(100),t=1):
    err = 2 - w.T@Z
    err = np.maximum(0,err)
    # HInge loss
    derr = np.mean( -Z , axis=1)
    # derr = ( -2*Z @ err.T )/Z.shape[1]
    # d2errz = 2* ( w @ w.T ) * np.sum(np.sign(err),axis=1)
    # eig = np.linalg.eigvals(d2errz/Z.shape[1])
    # Square loss
    sumg = 0
    # print(Z.shape[1],num.shape)
    for i in range(Z.shape[1]):
       sumg+= - (1-w.T@Z[:,i]) * Z[:,i]
    dsqrr = sumg/Z.shape[1]
    # dsqrr = (- (1 - (w.T@(Z)))@Z.T) / Z.shape[1]
    return np.mean(err),dsqrr.reshape(-1,1) #derr.reshape(-1,1)

# estimate epsion 
def estimate_epsilon(X):
    n = X.shape[0]
    est = 0
    for _ in range(1000):
        i = np.random.randint(n)
        j = np.random.randint(n)
        est += np.linalg.norm(X[i].toarray()-X[j].toarray())**2
    
    return est/1000


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

# # Clusters Based Buffer update policy
# def SF_buffer(C,B,x,k,num=np.ones(100),eps = 1):
    
#     diff = np.sum((C-x)**2,0)
#     if np.min(diff) >= eps and C.shape[1] < k:
#       C = np.c_[C,x]
#       B = np.c_[B,x]

#     else: 
#       J = np.argmin(diff)
#       if np.random.binomial(1,1/num[J]) == 1: 
#         B[:,[J]] = x
#       C[:,[J]] = C[:,[J]] + 0.05 * (C[:,[J]] - x) 
#       num[J] +=1
#     # if Bc.shape[1] >=2:print(Bc.shape[1])
#     return C,B,num

def SF_buffer(C,B,x,k,num=np.ones(100),eps = 1):
    
    diff = np.sum((C-x)**2,0)
    if np.min(diff) >= eps and C.shape[1] < k:
        C = np.c_[C,x]
        B.append(x)
    else: 
        J = np.argmin(diff)
        if np.random.binomial(1,1/num[J]) == 1 or B[J].shape[1]<=1: 
            if B[J].shape[1]<=1:
                B[J] = np.c_[B[J], x]
            else:
               B[J][:,[np.random.randint(B[J].shape[1])]] = x
        C[:,[J]] = C[:,[J]] + 0.05 * (C[:,[J]] - x) 
        # num[J] +=1
    return C,B,num


class options:
  eta = None
  lmd = None
  epoch = None

# buffer-based OGD for Pairwise leraning with SVM loss

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
  # auc = roc_auc_score(Y_test,1/(1+np.exp(-X_test.dot(w))))
  # AUCs_YIMING.append(auc)
  x_p = X[np.where(Y==1)[0]][0] ; x_p = x_p.toarray()
  x_n = X[np.where(Y==-1)[0]][0]  ; x_n = x_n.toarray()
  Bp = x_p.T
  Bn = x_n.T
  tp = 1
  tn = 1
  for k in range(options.epoch):
    i = k #np.random.randint(n)
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
#     g = np.mean(-(Dt - Dt @ np.diagflat(w.T@(Dt))),1,keepdims=1) + lmd * w
    ls , g = grad(w,Dt)
    w = w - eta *  g
    w = np.sign(w) * np.maximum(np.abs(w) - lmd * eta , 0 )
#     loss.append( np.mean((1 - w.T@(Dt))**2,axis=1) )
    loss.append(ls)
    # w = w/np.linalg.norm(w)
    if k%50 == 0:
      Tims_Y.append(time()-st)
      auc = roc_auc_score(Y_test, 1 / (  1 +  np.clip( np.exp(-X_test.dot(w)) ,0.001,10  )    ))
      AUCs_YIMING.append(auc)
  # pred_AUC = 1/(1+np.exp(-X_test.dot(w)))
  return np.array(Tims_Y),np.array(AUCs_YIMING),loss

# grid search engine
def gridsearch1(X_train,Y_train,s,kar=False):
  optLB = options()
  optLB.s = s

  b = 0
  n,d = X_train.shape
  split = n - int(n/4)
  for i in range(-5,-3):
    eta = 2**(i)
    for j in range(-9,-6):
      lmd = 10**j
      optLB.eta = eta
      optLB.lmd = lmd
      if X_train[:split,:].shape[0] <= 2000:
        optLB.epoch = X_train[:split,:].shape[0]
      else:
        optLB.epoch = 2000
      T,A,w = LimBuffer(X_train[:split,:],Y_train[:split],X_train[split:,:],Y_train[split:],optLB,kar)
      if (A[-1]) > b: 
        b = A[-1]
        eta_best = eta
        lmd_best = lmd
  return eta_best,lmd_best

# # Clusters Based OGD for Pairwise leraning with SVM loss

def Mixture(X,Y,X_test,Y_test,options):
  eta = options.eta
  eps = estimate_epsilon(X)
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
  # auc = roc_auc_score(Y_test,1/(1+np.exp(-X_test.dot(w))))
  # AUCs_YIMING.append(auc)
  x_p = X[np.where(Y==1)[0]][0] ; x_p = x_p.toarray()
  x_n = X[np.where(Y==-1)[0]][0]; x_n = x_n.toarray()
  Bp = x_p.T
  Bn = x_n.T
  BCp = Bp #;BCp = BCp[:, np.newaxis]
  BCn = Bn #;BCn = BCn[:, np.newaxis]
  Bp = [Bp] #;BCp = BCp[:, np.newaxis]
  Bn = [Bn] #;BCn = BCn[:, np.newaxis]
  numP = np.ones(s)
  numN = np.ones(s)
  tp = 1
  tn = 1
  for k in range(options.epoch):
    i = k#np.random.randint(n)
    x = X[[i],:] ; x = x.toarray()
    x = x.T
    if Y[i] == 1:
      Dt = [x - b for b in Bn] #x - Bn
      Dt = np.hstack(Dt)
    #   print(Dt.shape)
      BCp , Bp, numP = SF_buffer(BCp, Bp,x,s,numP,eps)
      ls , g = grad(w,Dt,numP,tp)
      tp +=1
    else:
      Dt = [b - x for b in Bp] #Bp - x
      Dt = np.hstack(Dt)
      BCn , Bn,numN = SF_buffer(BCn,Bn,x,s,numN,eps)
      ls , g = grad(w,Dt,numN,tn)
      tn+=1
    
    gp =  g #+ 0.1 * g1 #+ lmd * w
    # w = w - eta * gp
    # beta = 0.0
    ww = w - eta * gp
    w = np.sign(ww) * np.maximum(np.abs(ww) - lmd * eta , 0 )
#     loss.append( np.mean((1 - w.T@(Dt))**2,axis=1) )
    loss.append(ls)

    # w = w/np.linalg.norm(w)
    if k%50 == 0:
      Tims_Y.append(time()-st)
      auc = roc_auc_score(Y_test, 1 / (  1 +  np.clip( np.exp(-X_test.dot(w)) ,0.001,10  )    ))
      AUCs_YIMING.append(auc)
  # pred_AUC = 1/(1+np.exp(-X_test.dot(w)))
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



