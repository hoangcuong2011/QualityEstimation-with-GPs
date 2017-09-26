import sys
sys.path.append('../src')

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
#matplotlib inline 

from gpflow.likelihoods import Gaussian
from gpflow.kernels import RBF, White
from gpflow.mean_functions import Constant
from gpflow.sgpr import SGPR, GPRFITC
from gpflow.svgp import SVGP
from gpflow.gpr import GPR

from scipy.cluster.vq import kmeans2

from get_data import get_regression_data
from dgp import DGP
import time

from kgp.metrics import root_mean_squared_error as RMSE


X, Y, Xs, Ys = get_regression_data('kin8nm', split=0)
print 'N: {}, D: {}, Ns: {}'.format(X.shape[0], X.shape[1], Xs.shape[0])

def make_single_layer_models(X, Y, Z):
    D = X.shape[1]
    Y_mean, Y_std = np.average(Y), np.std(Y) 

    m_sgpr = SGPR(X, Y, RBF(D, variance=Y_std**2), Z.copy(), mean_function=Constant(Y_mean))
    m_svgp = SVGP(X, Y, RBF(D, variance=Y_std**2), Gaussian(), Z.copy(), mean_function=Constant(Y_mean))
    m_fitc = GPRFITC(X, Y, RBF(D, variance=Y_std**2), Z.copy(), mean_function=Constant(Y_mean))

    for m in [m_sgpr, m_svgp, m_fitc]:
        m.mean_function.fixed = True
        m.likelihood.variance = 0.1 * Y_std
    return m_sgpr, m_svgp, m_fitc

Z_100 = kmeans2(X, 10, minit='points')[0]
Z_500 = kmeans2(X, 5, minit='points')[0]
m_sgpr, m_svgp, m_fitc = make_single_layer_models(X, Y, Z_100)
m_sgpr_500, m_svgp_500, m_fitc_500 = make_single_layer_models(X, Y, Z_500)

if(1==1):
	k = RBF(input_dim=8, ARD=True)
	m = SGPR(np.array(X), np.array(Y), kern=k, Z=Z_100)
	m.likelihood.variance = 0.01
	m.optimize()
	mean, var = m.predict_y(Xs)
	rmse_predict = RMSE(Ys.reshape(-1,1), mean)
	print('Test RMSE Z_100:', rmse_predict)

	k = RBF(input_dim=8, ARD=True)
	m = SGPR(np.array(X), np.array(Y), kern=k, Z=Z_100)
	m.likelihood.variance = 0.01
	m.optimize()
	mean, var = m.predict_y(Xs)
	rmse_predict = RMSE(Ys.reshape(-1,1), mean)
	print('Test RMSE Z_500:', rmse_predict)


def make_dgp(X, Y, Z, L):
    D = X.shape[1]
    Y_mean, Y_std = np.average(Y), np.std(Y) 
    
    # the layer shapes are defined by the kernel dims, so here all hidden layers are D dimensional 
    kernels = []
    for l in range(L):
        kernels.append(RBF(D, lengthscales=1., variance=1.))
        
    # between layer noise (doesn't actually make much difference but we include it anyway)
    for kernel in kernels[:-1]:
        kernel += White(D, variance=1e-5) 
        
    mb = 10000 if X.shape[0] > 10000 else None 
    model = DGP(X, Y, Z, kernels, Gaussian(), num_samples=1, minibatch_size=mb)

    # same final layer inits we used for the single layer model
    model.layers[-1].kern.variance = Y_std**2
    model.likelihood.variance = Y_std*0.1 
    model.layers[-1].mean_function = Constant(Y_mean)
    model.layers[-1].mean_function.fixed = True
    
    # start the inner layers almost deterministically 
    for layer in model.layers[:-1]:
        layer.q_sqrt = layer.q_sqrt.value * 1e-5
    
    return model

m_dgp1 = make_dgp(X, Y, Z_100, 1)
m_dgp2 = make_dgp(X, Y, Z_100, 2)
m_dgp3 = make_dgp(X, Y, Z_100, 3)
m_dgp4 = make_dgp(X, Y, Z_100, 4)
m_dgp5 = make_dgp(X, Y, Z_100, 5)


def batch_assess(model, assess_model, X, Y):
    n_batches = 1
    lik, sq_diff = [], []
    for X_batch, Y_batch in zip(np.array_split(X, n_batches), np.array_split(Y, n_batches)):
        l, sq = assess_model(model, X_batch, Y_batch)
        lik.append(l)
        sq_diff.append(sq)
    lik = np.concatenate(lik, 0)
    sq_diff = np.array(np.concatenate(sq_diff, 0), dtype=float)
    return np.average(lik), np.average(sq_diff)**0.5

def assess_single_layer(model, X_batch, Y_batch):
    lik = model.predict_density(X_batch, Y_batch)
    mean, var = model.predict_y(X_batch)
    sq_diff = ((mean - Y_batch)**2)
    return lik, sq_diff 

S = 100
def assess_sampled(model, X_batch, Y_batch):
    lik = model.predict_density(X_batch, Y_batch, S)
    mean_samples, var_samples = model.predict_y(X_batch, 100)
    mean = np.average(mean_samples, 0)
    sq_diff = ((mean - Y_batch)**2)
    return lik, sq_diff



single_layer_models = [m_sgpr, m_svgp, m_fitc, m_sgpr_500, m_svgp_500, m_fitc_500]
single_layer_names = ['col sgp', 'sgp', 'fitc', 'col sgp 500', 'sgp 500', 'fitc 500']
for m, name in zip(single_layer_models, single_layer_names):
    t = time.time()
    m.optimize()
    lik, rmse = batch_assess(m, assess_single_layer, Xs, Ys)
    print '{:<16}  lik: {:.4f}, rmse: {:.4f}. Training time: {:.4f}'.format(name, lik, rmse, time.time() - t)



for m, name in zip([m_dgp1, m_dgp2, m_dgp3, m_dgp4, m_dgp5], ['dgp1 (sgp+adam)', 'dgp2', 'dgp3', 'dgp4', 'dgp5']):
    t = time.time()
    m.optimize(tf.train.AdamOptimizer(0.01), maxiter=5000)
    lik, rmse = batch_assess(m, assess_sampled, Xs, Ys)
    print '{:<16}  lik: {:.4f}, rmse: {:.4f}. Training time: {:.4f}'.format(name, lik, rmse, time.time() - t)