import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from time import time
from algorithm import *
from utils import *
from data_loader import *



# Parameters
dataname = 'a9a'
s = 4
epochs = 1
early_stop = 1000
path_to_data = 'path_to_data' 
# Load data
path = path_to_data + dataname
X,Y = get_data(path)
n,d = X.shape

## TRAIN AND TEST of Our algorithm
for i in range(epochs):
  kf=KFold(n_splits=2)
  for j, (train_index, test_index) in enumerate(kf.split(X)):
    X_train, X_test = X[train_index], X[test_index]
    Y_train, Y_test = Y[train_index], Y[test_index]
    if i ==0 and j ==0:
      eta,lmd = gridsearchM(X_train,Y_train,s)
      optM = options()
      optM.eta = eta
      optM.lmd = lmd
      optM.s = s
      optM.epoch = X_train.shape[0]
      if optM.epoch >= early_stop:
        optM.epoch = early_stop
    T_M,A_M,loss_M = Mixture(X_train,Y_train,X_test,Y_test,optM)


# plot results
fig = plt.figure()
ax = plt.axes()        
plt.rcParams.update({'font.size': 18})
ax.plot(T_M,A_M,label='FPOGD,s ='+str(s), color= 'b',linestyle='-',linewidth=3, markersize=5)
ax.set_ylabel('AUC')
ax.set_xlabel('time (s)')
ax.legend()
ax.grid(axis='both', which='both')
ax.set_title(dataname)
plt.ticklabel_format(axis='y', style='sci', scilimits=(-1,-1))
# plt.savefig(dataname+'_Cluster.pdf',  bbox_inches='tight')