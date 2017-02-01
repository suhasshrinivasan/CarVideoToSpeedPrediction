import json
from keras.callbacks import EarlyStopping, TensorBoard
from keras.models import Sequential
from keras.layers import Dense, GRU, LSTM, SimpleRNN, Convolution2D, MaxPooling2D
from keras.layers.core import Activation, Dropout, Flatten, Reshape
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l1l2
import matplotlib
import numpy as np
import os
import pdb
from sklearn.metrics import mean_squared_error
import random
import sys
import time
import warnings

from frames_to_features import extract_features
from video_to_frames import get_video_frames
from model_testing import hp_sweep, init_model
from smooth_signal import ma_smoothing, ewma_smoothing

matplotlib.use('qt5agg')
import matplotlib.pyplot as plt

# Initial Setup
model_type = 'cnn'
smooth_signal = ma_smoothing
data_smoothing_window_size = 1
show_model_plots = False
test_data_location = 'end'

using = 'cpu'
if len(sys.argv) == 2:
    if sys.argv[1] == '-g' or sys.argv[1] == '--gpu-optimized':
        using = 'gpu'
model_type = str.lower(model_type)
data_dir = 'data/'
filename_base = 'drive'
data_filename_base = os.path.join(data_dir, filename_base)
warnings.filterwarnings(action='ignore', module='scipy', message='^internal gelsd')

# Get feature data
data_filepath = data_filename_base + '.mp4'
num_frames = get_video_frames(data_filepath, override_existing=False)
extract_features(data_filepath, num_frames, extraction_network='resnet50', override_existing=False)
npz_file = np.load(data_filename_base + '.npz')
X = npz_file['arr_0']
print(X.shape)

# Get labels
with open (data_filename_base + '.json', 'r') as json_raw_data:
    time_speed_data = json_raw_data.readlines()[0]
time_speed_data = np.array(json.loads(time_speed_data))

# X_max = np.apply_over_axes(np.max, X, (1, 2))
# X_max = X_max.reshape((X_max.shape[0], X_max.shape[-1]))

# Split data
assert(X.shape[0] == time_speed_data.shape[0])
train_fraction = 0.9
num_train = int(X.shape[0] * train_fraction)
num_test = X.shape[0] - num_train

if test_data_location == 'end':
    X_train = X[:num_train,:]
    X_test = X[num_train:,:]
    time_speed_data_train = time_speed_data[:num_train,:]
    time_speed_data_test = time_speed_data[num_train:,:]
else:
    X_train = X[num_test:,:]
    X_test = X[:num_test,:]
    time_speed_data_train = time_speed_data[num_test:,:]
    time_speed_data_test = time_speed_data[:num_test,:]

time_train = time_speed_data_train[:,0].reshape(-1,1)
y_train = time_speed_data_train[:,1].reshape(-1,1)
time_test = time_speed_data_test[:,0].reshape(-1,1)
y_test = time_speed_data_test[:,1].reshape(-1,1)

# Additional preprocessing: row-wise differences
if data_smoothing_window_size > 1:
    X_train = ma_smoothing(X_train, data_smoothing_window_size)
    X_test = ma_smoothing(X_test, data_smoothing_window_size)
print('Data processed.')

# Predict off Extracted Features
if model_type[-2:] != 'nn':  # Simple Model
    # best_config = {'model_type': 'ridge', 'alpha': 3000.0}  # 8.44 MSE
    # best_config = {'model_type': 'ridge', 'alpha': 400.0}  # 8.44 MSE
    best_config = None
    if best_config is None:
        best_config, best_train_val_error = hp_sweep(model_type, X_train, y_train, X_test, y_test)
    model = init_model(best_config)
    model.fit(X_train, y_train)

    for smoothing_window_size in [1, 5, 9, 19, 39, 79, 119, 159]:
        print(smoothing_window_size)

        y_train_pred = model.predict(X_train)
        print(str.upper(model_type) +
            ' Model Train MSE: %.2f' % mean_squared_error(y_train, y_train_pred))

        y_train_pred_smoothed = smooth_signal(y_train_pred, smoothing_window_size)
        print(str.upper(model_type) +
            ' Model Smoothed Train MSE: %.2f' % mean_squared_error(y_train, y_train_pred_smoothed))

        y_test_pred = model.predict(X_test)
        print(str.upper(model_type) +
            ' Model Test MSE: %.2f' % mean_squared_error(y_test, y_test_pred))

        y_test_pred_smoothed = smooth_signal(y_test_pred, smoothing_window_size)
        print(str.upper(model_type) +
            ' Model Smoothed Test MSE: %.2f' % mean_squared_error(y_test, y_test_pred_smoothed))

        if show_model_plots:
            plt.plot(y_train,label='Actual')
            plt.plot(y_train_pred, label='Predicted')
            plt.plot(y_train_pred_smoothed, label='Predicted Smoothed')
            plt.show()

            plt.plot(y_test,label='Actual')
            plt.plot(y_test_pred, label='Predicted')
            plt.plot(y_test_pred_smoothed, label='Predicted Smoothed')
            plt.show()

else:  # Neural Network Model
    print('Keras model training optimized for ' + using + '.')
    # X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
    # X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))

    l1_reg = 0.0
    l2_reg = 0.0
    dropout_W = 0.0
    dropout_U = 0.0
    bn = True
    nb_epochs = 10
    batch_size = 100
    go_backwards = True
    validation_split = 0.05
    patience = 10
    num_hidden_units = 64

    nn_config = (l1_reg, l2_reg, dropout_W, dropout_U, bn, nb_epochs, batch_size, go_backwards)
    nn_config_str = '/l1_reg=' + str(l1_reg) + '/l2_reg=' + str(l2_reg) + '/dropout_W=' + str(dropout_W)\
        + '/dropout_U=' + str(dropout_U) + '/bn=' + str(bn) + '/nb_epochs=' + str(nb_epochs)\
        + '/batch_size=' + str(batch_size) + '/go_backwards=' + str(go_backwards)\
        + '/validation_split=' + str(validation_split) + '/patience=' + str(patience)\
        + '/num_hidden_units=' + str(num_hidden_units) + '/'\
        + str(random.randint(1, 1000000))
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=patience, verbose=0, mode='auto'),
        TensorBoard(log_dir='./logs' + nn_config_str, histogram_freq=0,
            write_graph=True, write_images=False),
    ]
    print(nn_config_str)

    # Compile and Train Neural Network Model
    model = Sequential()
    # model.add(GRU(100, unroll=True, consume_less=using,
    #     input_dim=X_train.shape[-1], input_length=1, go_backwards=go_backwards,
    #     W_regularizer=l1l2(l1=l1_reg, l2=l2_reg), U_regularizer=l1l2(l1=l1_reg, l2=l2_reg),
    #     b_regularizer=l1l2(l1=l1_reg, l2=l2_reg), dropout_W=dropout_W, dropout_U=dropout_U))
    # model.add(Dense(
    #     num_hidden_units,
    #     activation='relu',
    #     W_regularizer=l1l2(l1=l1_reg, l2=l2_reg),
    #     b_regularizer=l1l2(l1=l1_reg, l2=l2_reg),
    #     input_dim=X_train.shape[-1],
    # ))
    model.add(Convolution2D(
        num_hidden_units, 4, 4,
        border_mode='valid',
        subsample=(2, 2),
        input_shape=(8, 8, 2048)
    ))
    if bn:
        model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(3, 3)))
    if dropout_U > 0:
        model.add(Dropout(dropout_U))
    model.add(Flatten())
    model.add(GRU(1, unroll=True, consume_less=using,
        input_dim=X_train.shape[-1], input_length=1, go_backwards=go_backwards,
        W_regularizer=l1l2(l1=l1_reg, l2=l2_reg), U_regularizer=l1l2(l1=l1_reg, l2=l2_reg),
        b_regularizer=l1l2(l1=l1_reg, l2=l2_reg), dropout_W=dropout_W, dropout_U=dropout_U))
    # model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')

    t = time.time()
    hist = model.fit(X_train, y_train, nb_epoch=nb_epochs, batch_size=batch_size,
        validation_split=validation_split, verbose=2, callbacks=callbacks)
    print('Training Time: %.2f' % (time.time()-t))

    # Test Neural Network Model
    print(nn_config_str)

    for smoothing_window_size in [1, 49, 99, 149]:
        print(smoothing_window_size)

        y_train_pred = model.predict(X_train)
        print(str.upper(model_type) +
            ' Model Train MSE: %.2f' % mean_squared_error(y_train, y_train_pred))

        y_train_pred_smoothed = smooth_signal(y_train_pred, smoothing_window_size)
        print(str.upper(model_type) +
            ' Model Smoothed Train MSE: %.2f' % mean_squared_error(y_train, y_train_pred_smoothed))

        y_test_pred = model.predict(X_test)
        print(str.upper(model_type) +
            ' Model Test MSE: %.2f' % mean_squared_error(y_test, y_test_pred))

        y_test_pred_smoothed = smooth_signal(y_test_pred, smoothing_window_size)
        print(str.upper(model_type) +
            ' Model Smoothed Test MSE: %.2f' % mean_squared_error(y_test, y_test_pred_smoothed))

        if show_model_plots:
            plt.plot(y_train,label='Actual Train')
            plt.plot(y_train_pred, label='Predicted')
            plt.plot(y_train_pred_smoothed, label='Predicted Smoothed' + str(smoothing_window_size))
            plt.show()

            plt.plot(y_test,label='Actual Train')
            plt.plot(y_test_pred, label='Predicted')
            plt.plot(y_test_pred_smoothed, label='Predicted Smoothed' + str(smoothing_window_size))
            plt.show()
