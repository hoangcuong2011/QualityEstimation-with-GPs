# Author: Vincent Dubourg <vincent.dubourg@gmail.com>
#         Jake Vanderplas <vanderplas@astro.washington.edu>
# Licence: BSD 3 clause

import copy

import numpy as np
from sklearn.gaussian_process import GaussianProcess
from matplotlib import pyplot as pl
import GPflow
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

from GPflow.likelihoods import Gaussian
from GPflow.kernels import RBF, White
from GPflow.mean_functions import Constant
from GPflow.sgpr import SGPR, GPRFITC
from GPflow.svgp import SVGP
from GPflow.gpr import GPR

from scipy.cluster.vq import kmeans2

# from get_data import get_regression_data
# from dgp import DGP
import time

import csv

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

if(1==1):
	start = time.time()
	k = GPflow.kernels.RBF(input_dim=17, ARD=True)
	Z_100 = kmeans2(np.array(X), 1000, minit='points')[0]
	m = GPflow.sgpr.SGPR(np.array(X), np.array(Y), kern=k, Z=Z_100)
	m.likelihood.variance = 0.01
	m.optimize()
	mean, var = m.predict_y(X_test)
	rmse_predict = RMSE(y_test.reshape(-1,1), mean)
	#print(y_test.reshape(-1, 1))
	#print(mean)
	#print(var)


	with open("mean", "wb") as f:
		writer = csv.writer(f)
    		writer.writerows(mean)
	with open("var", "wb") as f:
	    	writer = csv.writer(f)
		writer.writerows(var)
	with open("predict_y", "wb") as f:
		writer = csv.writer(f)
		writer.writerows(y_test.reshape(-1,1))


	print('Test RMSE:', rmse_predict)
	end = time.time()

	print("Time execution", (end - start))

	dataset = np.loadtxt("testdata.txt.en_de", delimiter=",")


	X_test = dataset[:,0:17]

	y_test = dataset[:,17]


	X_train, X_test, X_valid = standardize_data(X_train_root, X_test, X_valid_root)

	mean, var = m.predict_y(X_test)
	rmse_predict = RMSE(y_test.reshape(-1,1), mean)
	#print(y_test.reshape(-1, 1))
	#print(mean)

	print('Test RMSE:', rmse_predict)
	end = time.time()

	print("Time execution", (end - start))