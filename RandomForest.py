
# Author: Vincent Dubourg <vincent.dubourg@gmail.com>
#         Jake Vanderplas <vanderplas@astro.washington.edu>
# Licence: BSD 3 clause

import copy

import numpy as np
from sklearn.gaussian_process import GaussianProcess
from matplotlib import pyplot as pl

from kgp.metrics import root_mean_squared_error as RMSE

from sklearn.gaussian_process import GaussianProcess
from matplotlib import pyplot as pl
import numpy as np
import sys
np.random.seed(42)

# Keras
from keras.models import Model
from keras.layers import Input, Dense, Dropout
from keras.optimizers import RMSprop, Adam
from keras.callbacks import EarlyStopping

from kgp.utils.experiment import train

# Metrics
from kgp.metrics import root_mean_squared_error as RMSE

import numpy as np
import tensorflow as tf


from scipy.cluster.vq import kmeans2

#from get_data import get_regression_data
from dgp import DGP
import time

import time

import numpy as np

from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import learning_curve
from sklearn.kernel_ridge import KernelRidge
import matplotlib.pyplot as plt


from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor



def standardize_data(X_train, X_test, X_valid):
    X_mean = np.mean(X_train, axis=0)
    X_std = np.std(X_train, axis=0)

    X_train -= X_mean
    X_train /= X_std
    X_test -= X_mean
    X_test /= X_std
    X_valid -= X_mean
    X_valid /= X_std

    return X_train, X_test, X_valid

np.random.seed(1)

#scikit-learn code

dataset = np.loadtxt("traindata.txt", delimiter=",") 
X_train = dataset[:,0:17]
y_train = dataset[:,17]


dataset = np.loadtxt("testdata.txt", delimiter=",")


X_test = dataset[:,0:17]
y_test = dataset[:,17]
dataset = np.loadtxt("validdata.txt", delimiter=",")


X_valid = dataset[:,0:17]
y_valid = dataset[:,17]


X_train_root = X_train

X_valid_root = X_valid

X_train, X_test, X_valid = standardize_data(copy.deepcopy(X_train_root), X_test, copy.deepcopy(X_valid_root))


    

X = X_train
Y = y_train.reshape(-1, 1)

Xs, Ys = X_test, y_test.reshape(-1, 1)


tuned_parameters = {'n_estimators': [500, 700, 1000], 'max_depth': [None, 1, 2, 3, 4], 'min_samples_split': [2]}


#clf = GridSearchCV(ensemble.RandomForestRegressor(), tuned_parameters, cv=5, scoring=r2_score, n_jobs=-1, verbose=1)

clf = GridSearchCV(RandomForestRegressor(), tuned_parameters, cv=5, n_jobs=-1, verbose=1)

#clf.fit(X_train, y_train)
    
clf.fit(X_train, y_train)


# regr_rf = RandomForestRegressor(max_depth=max_depth, random_state=2)
# regr_rf.fit(X_train, np.ravel(y_train.reshape(-1, 1)))

# Predict on new data



y_multirf = clf.predict(X_test)


rmse_predict = RMSE(y_test.reshape(-1,1), y_multirf)

print(rmse_predict)



dataset = np.loadtxt("testdata.txt.en_de", delimiter=",")


X_test = dataset[:,0:17]

y_test = dataset[:,17]


X_train, X_test, X_valid = standardize_data(X_train_root, X_test, X_valid_root)


y_multirf = clf.predict(X_test)


rmse_predict = RMSE(y_test.reshape(-1,1), y_multirf)

print(rmse_predict)

