
import numpy as np
from sklearn.gaussian_process import GaussianProcess
from matplotlib import pyplot as pl

from kgp.metrics import root_mean_squared_error as RMSE

from sklearn.gaussian_process import GaussianProcess
from matplotlib import pyplot as pl
import numpy as np
import sys
import sklearn.tree
np.random.seed(42)
from operator import itemgetter, attrgetter

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



from sklearn import tree

param_grid = {'max_depth': [None, 1, 2, 3, 4], 'min_samples_split': [2, 3, 4, 5]}

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


dataset = np.loadtxt("validdata.txt", delimiter=",")


X_valid = dataset[:,0:17]
y_valid = dataset[:,17]

dataset = np.loadtxt("testdata.txt", delimiter=",")


X_test = dataset[:,0:17]

y_test = dataset[:,17]


X_train, X_test, X_valid = standardize_data(X_train, X_test, X_valid)


X = X_train
y = y_train.reshape(-1, 1)



dt = sklearn.tree.DecisionTreeRegressor()


grid_search = GridSearchCV(dt, param_grid=param_grid, cv=5)

from time import time

start = time()

grid_search.fit(X, y)

def report(grid_scores, n_top=3):
    """Report top n_top parameters settings, default n_top=3.

    Args
    ----
    grid_scores -- output from grid or random search
    n_top -- how many to report, of top models

    Returns
    -------
    top_params -- [dict] top parameter settings found in
                  search
    """
    top_scores = sorted(grid_scores,
                        key=itemgetter(1),
                        reverse=True)[:n_top]
    for i, score in enumerate(top_scores):
        print("Model with rank: {0}".format(i + 1))
        print(("Mean validation score: "
               "{0:.3f} (std: {1:.3f})").format(
               score.mean_validation_score,
               np.std(score.cv_validation_scores)))
        print("Parameters: {0}".format(score.parameters))
        print("")

    return top_scores[0].parameters


print(("\nGridSearchCV took {:.2f} "
           "seconds for {:d} candidate "
           "parameter settings.").format(time() - start,
                len(grid_search.grid_scores_)))

top_params = report(grid_search.grid_scores_, 3)
print(top_params)




y_multirf = grid_search.predict(X_test)


rmse_predict = RMSE(y_test.reshape(-1,1), y_multirf)

print(rmse_predict)



dataset = np.loadtxt("testdata.txt.en_de", delimiter=",")


X_test = dataset[:,0:17]

y_test = dataset[:,17]


X_train, X_test, X_valid = standardize_data(X_train, X_test, X_valid)


y_multirf = grid_search.predict(X_test)


rmse_predict = RMSE(y_test.reshape(-1,1), y_multirf)

print(rmse_predict)


