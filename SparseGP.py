# Author: Vincent Dubourg <vincent.dubourg@gmail.com>
#         Jake Vanderplas <vanderplas@astro.washington.edu>
# Licence: BSD 3 clause

import numpy as np
from sklearn.gaussian_process import GaussianProcess
from matplotlib import pyplot as pl
import gpflow
import time

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

# from get_data import get_regression_data
# from dgp import DGP
import time



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

X_train, X_test, X_valid = standardize_data(X_train, X_test, X_valid)
    


X = X_train
Y = y_train.reshape(-1, 1)	
#print(Y)
#print(len(Y))

#k = gpflow.kernels.Matern52(input_dim=17, ARD=True)
if(1==1):
	start = time.time()
	k = gpflow.kernels.RBF(input_dim=17, ARD=True)
	Z_100 = kmeans2(np.array(X), 1, minit='points')[0]
	#print(Z_100)
	m = gpflow.sgpr.SGPR(np.array(X), np.array(Y), kern=k, Z=Z_100)
	m.likelihood.variance = 0.01
	m.optimize()
	mean, var = m.predict_y(X_test)
	rmse_predict = RMSE(y_test.reshape(-1,1), mean)
	print('Test RMSE:', rmse_predict)
	end = time.time()

	print("Time execution", (end - start))


if(1==1):
	start = time.time()
	k = gpflow.kernels.RBF(input_dim=17, ARD=True)
	Z_100 = kmeans2(np.array(X), 10, minit='points')[0]
	#print(Z_100)
	m = gpflow.sgpr.SGPR(np.array(X), np.array(Y), kern=k, Z=Z_100)
	m.likelihood.variance = 0.01
	m.optimize()
	mean, var = m.predict_y(X_test)
	rmse_predict = RMSE(y_test.reshape(-1,1), mean)
	print('Test RMSE:', rmse_predict)
	end = time.time()

	print("Time execution", (end - start))



if(1==1):
	start = time.time()
	k = gpflow.kernels.RBF(input_dim=17, ARD=True)
	Z_100 = kmeans2(np.array(X), 100, minit='points')[0]
	#print(Z_100)
	m = gpflow.sgpr.SGPR(np.array(X), np.array(Y), kern=k, Z=Z_100)
	m.likelihood.variance = 0.01
	m.optimize()
	mean, var = m.predict_y(X_test)
	rmse_predict = RMSE(y_test.reshape(-1,1), mean)
	print('Test RMSE:', rmse_predict)
	end = time.time()

	print("Time execution", (end - start))




# Plot the function, the prediction and the 95% confidence interval based on
# the MSE
#fig = pl.figure()
#pl.plot(x, f(x), 'r:', label=u'$f(x) = x\,\sin(x)$')
#pl.plot(X, y, 'r.', markersize=10, label=u'Observations')
#pl.plot(x, y_pred, 'b-', label=u'Prediction')
#pl.fill(np.concatenate([x, x[::-1]]),
#        np.concatenate([y_pred - 1.9600 * sigma,
#                       (y_pred + 1.9600 * sigma)[::-1]]),
#        alpha=.5, fc='b', ec='None', label='95% confidence interval')
#pl.xlabel('$x$')
#pl.ylabel('$f(x)$')
#pl.ylim(-10, 20)
#pl.legend(loc='upper left')

#pl.show()
