
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
from sklearn.preprocessing import normalize
from random import shuffle



def get_data(filePath):
    data = load_svmlight_file(filePath)
    X = data[0]; Y = data[1]
    X = 4* normalize(X, norm='l2', axis=1)
    Y = Y.reshape(-1,1)
    print('data Loaded and Verified')
    #print(np.sum(Y==-1) / np.sum(Y==1))
    # Convert to Binary
    if max(Y)>1:
        YY = np.ones_like(Y)
        yy = []
        for i in range(5):
            YY[Y==i] = -1
        Y = YY
    # Shuffle
    indices = np.arange(X.shape[0]) 
    shuffle(indices)
    X = X[list(indices)] 
    Y = Y[list(indices)] 
    return X,Y
