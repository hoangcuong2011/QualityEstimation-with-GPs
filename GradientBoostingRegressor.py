from sklearn.ensemble import GradientBoostingRegressor
from sklearn.utils.estimator_checks import check_estimator
from skbayes.rvm_ard_models import RegressionARD,ClassificationARD,RVR,RVC
from sklearn.svm import SVR
from sklearn.grid_search import GridSearchCV
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import time
from sklearn.metrics import mean_squared_error


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


# train rvr
rvm = GridSearchCV( GradientBoostingRegressor(), 
                    param_grid = {'learning_rate': [1e-3,1e-1,1],
                                  'max_depth': [None, 1, 2, 3, 4],
                                  'n_estimators':[500, 700, 1000]})


#rvm = RVR(gamma = 1,kernel = 'rbf')
t1 = time.time()
rvm.fit(X_train,y_train)
t2 = time.time()
y_hat     = rvm.predict(X_test)
#rvm_err   = mean_squared_error(y_hat,y_test)
#rvs       = np.sum(rvm.active_)
#print "RVM error on test set is {0}, number of relevant vectors is {1}, time {2}".format(rvm_err, rvs, t2 - t1)

rmse_predict = RMSE(y_test.reshape(-1,1), y_hat)

print(rmse_predict)




dataset = np.loadtxt("testdata.txt.en_de", delimiter=",")


X_test = dataset[:,0:17]

y_test = dataset[:,17]


X_train, X_test, X_valid = standardize_data(X_train_root, X_test, X_valid_root)


y_hat     = rvm.predict(X_test)

rmse_predict = RMSE(y_test.reshape(-1,1), y_hat)

print(rmse_predict)

