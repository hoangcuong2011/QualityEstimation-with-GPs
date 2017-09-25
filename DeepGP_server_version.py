# Author: Vincent Dubourg <vincent.dubourg@gmail.com>
#         Jake Vanderplas <vanderplas@astro.washington.edu>
# Licence: BSD 3 clause

import numpy as np
from sklearn.gaussian_process import GaussianProcess
from matplotlib import pyplot as pl
import GPflow

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

#from get_data import get_regression_data
from dgp import DGP
import time



def batch_assess(model, assess_model, X, Y):
    n_batches = 1
    lik, sq_diff = [], []
    l, sq = assess_model(model, X, Y)
    lik.append(l)
    sq_diff.append(sq)

    #print(sq_diff, lik)
        
    return np.average(lik), np.average(sq_diff)**0.5

def assess_single_layer(model, X_batch, Y_batch):
    lik = model.predict_density(X_batch, Y_batch)
    mean, var = model.predict_y(X_batch)
    sq_diff = ((mean - Y_batch)**2)
    return lik, sq_diff 

def assess_sampled(model, X_batch, Y_batch):
    lik = model.predict_density(X_batch, Y_batch, 10)
    mean_samples, var_samples = model.predict_y(X_batch, 1)
    mean = np.average(mean_samples, 0)
    sq_diff = ((mean - Y_batch)**2)
    return lik, sq_diff

def make_dgp(X, Y, Z, L):
    
    Y_mean, Y_std = np.average(Y), np.std(Y) 
    
    # the layer shapes are defined by the kernel dims, so here all hidden layers are D dimensional 
    kernels = []
    for l in range(L):
        kernels.append(RBF(input_dim=17, ARD=True))
        
    mb = 128 if X.shape[0] > 128 else None 
    model = DGP(X, Y, Z, kernels, Gaussian(), num_samples=1, minibatch_size=mb)

    # same final layer inits we used for the single layer model
    #model.layers[-1].kern.variance = Y_std**2
    model.likelihood.variance = 0.01 
    #model.layers[-1].mean_function = Constant(Y_mean)
    #model.layers[-1].mean_function.fixed = True
    
    # start the inner layers almost deterministically 
    for layer in model.layers[:-1]:
        layer.q_sqrt = layer.q_sqrt.value * 1e-5
    
    return model


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
X_train, X_test, X_valid = standardize_data(X_train, X_test, X_valid)
    

X = X_train
Y = y_train.reshape(-1, 1)	

Xs, Ys = X_test, y_test.reshape(-1, 1)




if(1==1):
	Z_100 = kmeans2(np.array(X), 1, minit='points')[0]

	Y_mean, Y_std = np.average(Y), np.std(Y) 
	k = RBF(input_dim=17, ARD=True)
	m = SGPR(np.array(X), np.array(Y), kern=k, Z=Z_100)
	m.likelihood.variance = 0.01
	m.optimize()
	mean, var = m.predict_y(X_test)
	rmse_predict = RMSE(y_test.reshape(-1,1), mean)
	print('Test RMSE:', rmse_predict)

	m_dgp1 = make_dgp(X, Y, Z_100, 1)
	m_dgp2 = make_dgp(X, Y, Z_100, 2)
	m_dgp3 = make_dgp(X, Y, Z_100, 3)
	m_dgp4 = make_dgp(X, Y, Z_100, 4)
	m_dgp5 = make_dgp(X, Y, Z_100, 5)


	for m, name in zip([m_dgp1, m_dgp2, m_dgp3, m_dgp4, m_dgp5], ['dgp1 (sgp+adam)', 'dgp2', 'dgp3', 'dgp4', 'dgp5']):
	    t = time.time()
	    m.optimize(tf.train.AdamOptimizer(0.01), maxiter=5000)
	    lik, rmse = batch_assess(m, assess_sampled, Xs, Ys)
	    print('Test RMSE 1:', rmse, name)


if(1==1):
	Z_100 = kmeans2(np.array(X), 10, minit='points')[0]

	Y_mean, Y_std = np.average(Y), np.std(Y) 
	k = RBF(input_dim=17, ARD=True)
	m = SGPR(np.array(X), np.array(Y), kern=k, Z=Z_100)
	m.likelihood.variance = 0.01
	m.optimize()
	mean, var = m.predict_y(X_test)
	rmse_predict = RMSE(y_test.reshape(-1,1), mean)
	print('Test RMSE:', rmse_predict)

	m_dgp1 = make_dgp(X, Y, Z_100, 1)
	m_dgp2 = make_dgp(X, Y, Z_100, 2)
	m_dgp3 = make_dgp(X, Y, Z_100, 3)
	m_dgp4 = make_dgp(X, Y, Z_100, 4)
	m_dgp5 = make_dgp(X, Y, Z_100, 5)


	for m, name in zip([m_dgp1, m_dgp2, m_dgp3, m_dgp4, m_dgp5], ['dgp1 (sgp+adam)', 'dgp2', 'dgp3', 'dgp4', 'dgp5']):
	    t = time.time()
	    m.optimize(tf.train.AdamOptimizer(0.01), maxiter=5000)
	    lik, rmse = batch_assess(m, assess_sampled, Xs, Ys)
	    print('Test RMSE 10:', rmse, name)



if(1==1):
	Z_100 = kmeans2(np.array(X), 100, minit='points')[0]

	Y_mean, Y_std = np.average(Y), np.std(Y) 
	k = RBF(input_dim=17, ARD=True)
	m = SGPR(np.array(X), np.array(Y), kern=k, Z=Z_100)
	m.likelihood.variance = 0.01
	m.optimize()
	mean, var = m.predict_y(X_test)
	rmse_predict = RMSE(y_test.reshape(-1,1), mean)
	print('Test RMSE:', rmse_predict)

	m_dgp1 = make_dgp(X, Y, Z_100, 1)
	m_dgp2 = make_dgp(X, Y, Z_100, 2)
	m_dgp3 = make_dgp(X, Y, Z_100, 3)
	m_dgp4 = make_dgp(X, Y, Z_100, 4)
	m_dgp5 = make_dgp(X, Y, Z_100, 5)


	for m, name in zip([m_dgp1, m_dgp2, m_dgp3, m_dgp4, m_dgp5], ['dgp1 (sgp+adam)', 'dgp2', 'dgp3', 'dgp4', 'dgp5']):
	    t = time.time()
	    m.optimize(tf.train.AdamOptimizer(0.01), maxiter=5000)
	    lik, rmse = batch_assess(m, assess_sampled, Xs, Ys)
	    print('Test RMSE 100:', rmse, name)




if(1==1):
	Z_100 = kmeans2(np.array(X), 1000, minit='points')[0]

	Y_mean, Y_std = np.average(Y), np.std(Y) 
	k = RBF(input_dim=17, ARD=True)
	m = SGPR(np.array(X), np.array(Y), kern=k, Z=Z_100)
	m.likelihood.variance = 0.01
	m.optimize()
	mean, var = m.predict_y(X_test)
	rmse_predict = RMSE(y_test.reshape(-1,1), mean)
	print('Test RMSE:', rmse_predict)

	m_dgp1 = make_dgp(X, Y, Z_100, 1)
	m_dgp2 = make_dgp(X, Y, Z_100, 2)
	m_dgp3 = make_dgp(X, Y, Z_100, 3)
	m_dgp4 = make_dgp(X, Y, Z_100, 4)
	m_dgp5 = make_dgp(X, Y, Z_100, 5)


	for m, name in zip([m_dgp1, m_dgp2, m_dgp3, m_dgp4, m_dgp5], ['dgp1 (sgp+adam)', 'dgp2', 'dgp3', 'dgp4', 'dgp5']):
	    t = time.time()
	    m.optimize(tf.train.AdamOptimizer(0.01), maxiter=5000)
	    lik, rmse = batch_assess(m, assess_sampled, Xs, Ys)
	    print('Test RMSE 100:', rmse, name)



