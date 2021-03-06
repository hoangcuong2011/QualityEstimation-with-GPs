"""
MLP regression on Kin40k data.
Attains about 0.14 RMSE within 150 epochs.
Reference: https://arxiv.org/abs/1511.02222
"""
from __future__ import print_function

import copy
import numpy as np
import sys
np.random.seed(42)

# Keras
from keras.models import Model
from keras.layers import Input, Dense, Dropout
from keras.optimizers import RMSprop, Adam
from keras.callbacks import EarlyStopping

print(sys.path)

# Model assembling and executing
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


def assemble_mlp(input_shape, output_shape):
    """Assemble a simple MLP model.
    """
    inputs = Input(shape=input_shape)
    hidden = Dense(512, activation='relu', name='dense1')(inputs)
    hidden = Dropout(0.5)(hidden)
    hidden = Dense(512, activation='relu', name='dense2')(hidden)
    hidden = Dropout(0.5)(hidden)
    outputs = Dense(1, activation='linear')(hidden)
    return Model(inputs=inputs, outputs=outputs)


def main():
    # Load data
    
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


    print("MAX_Test ", np.amax(y_test))
    print("MAX_Train", np.amax(y_train))
    print("MAX_Valid", np.amax(y_valid))

    print("MIN_Test ", np.amin(y_test))
    print("MIN_Train", np.amin(y_train))
    print("MIN_Valid", np.amin(y_valid))

    UPPERBOUND = 1
    print(y_train)
    y_train = y_train/UPPERBOUND
    y_valid = y_valid/UPPERBOUND
    print(y_train)


    data = {
        'train': (X_train, y_train),
        'valid': (X_valid, y_valid),
        'test': (X_test, y_test),
    }
    # Model & training parameters
    input_shape = data['train'][0].shape[1:]
    output_shape = data['train'][1].shape[1:]
    batch_size = 128
    epochs = 250

    # Construct & compile the model
    model = assemble_mlp(input_shape, output_shape)
    model.compile(optimizer=Adam(1e-4), loss='mse')
    # model.load_weights('checkpoints/mlp_kin40k.h5', by_name=True)

    # Callbacks
    # callbacks = [
    #     EarlyStopping(monitor='val_loss', patience=10),
    # ]
    callbacks = []

    # Train the model
    history = train(model, data, callbacks=callbacks,
                    checkpoint='mlp_kin40k', checkpoint_monitor='val_loss',
                    epochs=epochs, batch_size=batch_size, verbose=1)

    # Test the model
    X_test, y_test = data['test']
    y_preds = model.predict(X_test)
    rmse_predict = RMSE(y_test, y_preds*UPPERBOUND)
    print('Test RMSE:', rmse_predict)


    dataset = np.loadtxt("testdata.txt.en_de", delimiter=",")


    X_test = dataset[:,0:17]

    y_test = dataset[:,17]

    X_train, X_test, X_valid = standardize_data(X_train_root, X_test, X_valid_root)

    y_preds = model.predict(X_test)

    rmse_predict = RMSE(y_test, y_preds*UPPERBOUND)

    print('Test Adaptation RMSE:', rmse_predict)


if __name__ == '__main__':
    main()
