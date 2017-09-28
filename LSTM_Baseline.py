"""
LSTM regression on Actuator data.
"""
from __future__ import print_function

import numpy as np
np.random.seed(42)

# Keras
from keras.optimizers import Adagrad, Adam, SGD, RMSprop
from keras.callbacks import EarlyStopping

# Dataset interfaces
from kgp.datasets.sysid import load_data
from kgp.datasets.data_utils import data_to_seq

# Model assembling and executing
from kgp.utils.assemble import load_NN_configs, assemble
from kgp.utils.experiment import train

# Metrics
from kgp.metrics import root_mean_squared_error as RMSE



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


def main():


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

    X_train = X_train.reshape(-1,17)

    X_test = X_test.reshape(-1,17)

    X_valid = X_valid.reshape(-1,17)

    y_valid = y_valid.reshape(-1, 1)

    y_test = y_test.reshape(-1, 1)

    y_train = y_train.reshape(-1, 1)


    X_train, y_train = data_to_seq(X_train, y_train, t_lag=32, t_future_shift=1, t_future_steps=1, t_sw_step=1)
    X_valid, y_valid = data_to_seq(X_valid, y_valid, t_lag=32, t_future_shift=1, t_future_steps=1, t_sw_step=1)
    X_test, y_test = data_to_seq(X_test, y_test, t_lag=32, t_future_shift=1, t_future_steps=1, t_sw_step=1)

    data = {
        'train': [X_train, y_train],
        'valid': [X_valid, y_valid],
        'test': [X_test, y_test],
    }

    # Model & training parameters
    input_shape = list(data['train'][0].shape[1:])
    output_shape = list(data['train'][1].shape[1:])
    batch_size =128 
    epochs = 250

    nn_params = {
        'H_dim': 512,
        'H_activation': 'tanh',
        'dropout': 0.5,
    }

    # Retrieve model config
    configs = load_NN_configs(filename='lstm.yaml',
                              input_shape=input_shape,
                              output_shape=output_shape,
                              params=nn_params)

    # Construct & compile the model
    model = assemble('LSTM', configs['1H'])
    model.compile(optimizer=Adam(1e-4), loss='mse')

    # Callbacks
    #callbacks = [EarlyStopping(monitor='val_nlml', patience=10)]
    callbacks = []

    # Train the model
    history = train(model, data, callbacks=callbacks,
                    checkpoint='lstm', checkpoint_monitor='val_loss',
                    epochs=epochs, batch_size=batch_size, verbose=1)

    # Test the model
    X_test, y_test = data['test']
    y_preds = model.predict(X_test)
    rmse_predict = RMSE(y_test, y_preds)
    print('Test RMSE:', rmse_predict)


if __name__ == '__main__':
    main()
