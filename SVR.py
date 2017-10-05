# Author: Vincent Dubourg <vincent.dubourg@gmail.com>
#         Jake Vanderplas <vanderplas@astro.washington.edu>
# Licence: BSD 3 clause
import copy

import numpy as np
from sklearn.gaussian_process import GaussianProcess
from matplotlib import pyplot as pl
import gpflow

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

from gpflow.likelihoods import Gaussian
from gpflow.kernels import RBF, White
from gpflow.mean_functions import Constant
from gpflow.sgpr import SGPR, GPRFITC
from gpflow.svgp import SVGP
from gpflow.gpr import GPR

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

dataset = np.loadtxt("validdata.txt", delimiter=",") 
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


svr = GridSearchCV(SVR(kernel='rbf', gamma=0.1), cv=5,
                   param_grid={"C": [1e0, 1e1, 1e2, 1e3, 1e-1, 1e-2, 1e-3],
                               "gamma": np.logspace(-2, 2, 5)})

kr = GridSearchCV(KernelRidge(kernel='rbf', gamma=0.1), cv=5,
                  param_grid={"alpha": [1e0, 0.1, 1e-2, 1e-3],
                              "gamma": np.logspace(-2, 2, 5)})

t0 = time.time()
svr.fit(X_train, np.ravel(y_train.reshape(-1, 1)))
svr_fit = time.time() - t0
print("SVR complexity and bandwidth selected and model fitted in %.3f s"
      % svr_fit)

t0 = time.time()

kr.fit(X_train, np.ravel(y_train.reshape(-1, 1)))

kr_fit = time.time() - t0
print("KRR complexity and bandwidth selected and model fitted in %.3f s"
      % kr_fit)

sv_ratio = svr.best_estimator_.support_.shape[0] / train_size
print("Support vector ratio: %.3f" % sv_ratio)

t0 = time.time()
y_svr = svr.predict(X_test)
rmse_predict = RMSE(y_test.reshape(-1,1), y_svr)

print(rmse_predict)
svr_predict = time.time() - t0
print("SVR prediction for %d inputs in %.3f s"
      % (X_test.shape[0], svr_predict))

t0 = time.time()
y_kr = kr.predict(X_test)
kr_predict = time.time() - t0
print("KRR prediction for %d inputs in %.3f s"
      % (X_test.shape[0], kr_predict))

rmse_predict = RMSE(y_test.reshape(-1,1), y_kr)

print(rmse_predict)






dataset = np.loadtxt("testdata.txt.en_de", delimiter=",")


X_test = dataset[:,0:17]

y_test = dataset[:,17]


X_train, X_test, X_valid = standardize_data(X_train_root, X_test, X_valid_root)




t0 = time.time()
y_svr = svr.predict(X_test)
rmse_predict = RMSE(y_test.reshape(-1,1), y_svr)

print(rmse_predict)
svr_predict = time.time() - t0
print("SVR prediction for %d inputs in %.3f s"
      % (X_test.shape[0], svr_predict))

t0 = time.time()
y_kr = kr.predict(X_test)
kr_predict = time.time() - t0
print("KRR prediction for %d inputs in %.3f s"
      % (X_test.shape[0], kr_predict))

rmse_predict = RMSE(y_test.reshape(-1,1), y_kr)

print(rmse_predict)
