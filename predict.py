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
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
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

# A Few Important Configurable Variables
model_type = 'ridge'
extraction_network = 'resnet50'
smooth_signal = ma_smoothing
smooth_data = ma_smoothing
smooth_data_window_size = 1  # TODO: Remove this feature?
scale_data = True
best_pca_n_components = 0  # <= num features, 0 for No PCA, None to Hyperparameter Sweep
train_fraction = 0.839  # Split chosen based on ratio of highway to city driving in data
k_fold = 10  # For Cross-Validation. Needs to be >5 for training folds to have enough data
show_model_plots = True
test_data_location = 'end'
# best_config = None
best_config = {'model_type': 'ridge', 'alpha': 4000.0}  # 8.44 MSE
print(best_pca_n_components)

# Initial Setup
using = 'cpu'
if len(sys.argv) >= 2 and (sys.argv[-1] == '-g' or sys.argv[1] == '--gpu-optimized'):
    using = 'gpu'
model_type = str.lower(model_type)
data_dir = 'data/'
filename_base = 'drive'
data_filename_base = os.path.join(data_dir, filename_base)
features_filepath = data_filename_base + '_' + extraction_network + '.npz'
warnings.filterwarnings(action='ignore', module='scipy', message='^internal gelsd')
print ('Set to train ' + str.upper(model_type) + ' Model.')

# Get feature data
data_filepath = data_filename_base + '.mp4'
num_frames = get_video_frames(data_filepath, override_existing=False)
num_frames = 3
extract_features(data_filepath, num_frames, extraction_network=extraction_network, override_existing=False)
npz_file = np.load(features_filepath)
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
if smooth_data_window_size > 1:
    X_train = smooth_data(X_train, smooth_data_window_size)
    X_test = smooth_data(X_test, smooth_data_window_size)
print('Data processed.')

# Predict off Extracted Features
if model_type[-2:] != 'nn':  # Simple Model
    if best_config is None:
        if best_pca_n_components is None:
            pca_n_components_list = [160, 320, 640, 1280, 2048]
            # pca_n_components_list = [240, 480, 960, 1280, 1660]
        else:
            pca_n_components_list = [best_pca_n_components]
        best_config, best_train_val_error = hp_sweep(
            model_type, X_train, y_train, k_fold, scale_data, pca_n_components_list)

    # Scale Data
    if scale_data:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    # PCA on Data
    if best_pca_n_components != 0:
        pca = PCA(n_components=best_pca_n_components)
        X_train = pca.fit_transform(X_train)
        X_test = pca.transform(X_test)

    # Train Chosen Model
    model = init_model(best_config)
    model.fit(X_train, y_train)

    for smoothing_window_size in [1, 5, 9, 19, 39, 79, 119, 159, 199, 239]:
        print(smoothing_window_size)

        y_train_pred = model.predict(X_train)
        print(str.upper(model_type) +
            ' Model Train MSE: %.2f' % mean_squared_error(y_train, y_train_pred))

        y_train_pred_smoothed = smooth_signal(y_train_pred, smoothing_window_size)
        print(str.upper(model_type) +
            ' Model Smoothed Train MSE: %.2f' % mean_squared_error(y_train, y_train_pred_smoothed))

        y_test_pred = model.predict(X_test)
        print(y_test_pred.shape)
        # y_test = y_test[-862:]
        # y_test_pred = y_test_pred[-862:]
        # y_test = y_test[:862]
        # y_test_pred = y_test_pred[:862]
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
    print('Keras model training optimized for ' + str.upper(using) + '.')
    # X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
    # X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))

    l1_reg = 0.0
    l2_reg = 0.0
    dropout_W = 0.0
    dropout_U = 0.0
    bn = True
    nb_epochs = 100
    batch_size = 100
    go_backwards = True
    validation_split = 0.05
    patience = 10
    num_hidden_units = 16

    nn_config = (l1_reg, l2_reg, dropout_W, dropout_U, bn, nb_epochs, batch_size, go_backwards)
    nn_config_str = '/l1_reg=' + str(l1_reg) + '/l2_reg=' + str(l2_reg) + '/dropout_W=' + str(dropout_W)\
        + '/dropout_U=' + str(dropout_U) + '/bn=' + str(bn) + '/nb_epochs=' + str(nb_epochs)\
        + '/batch_size=' + str(batch_size) + '/go_backwards=' + str(go_backwards)\
        + '/validation_split=' + str(validation_split) + '/patience=' + str(patience)\
        + '/num_hidden_units=' + str(num_hidden_units) + '/'\
        + str(random.randint(1, 1000000))
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=patience, verbose=0, mode='auto'),
        # TensorBoard(log_dir='./logs' + nn_config_str, histogram_freq=0,
        #     write_graph=True, write_images=False),
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
        input_shape=(8, 8, 2048),
        W_regularizer=l1l2(l1_reg, l2_reg),
    ))
    if bn:
        model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(3, 3)))
    if dropout_U > 0:
        model.add(Dropout(dropout_U))
    # print(model.layers[-1].output_shape)
    # model.add(Reshape((None, 1, num_hidden_units), input_shape=(None, 1, 1, 64)))
    # print(model.layers[-1].output_shape)
    # model.add(GRU(1, unroll=True, consume_less=using,
    #     input_dim=X_train.shape[-1], input_length=1, go_backwards=go_backwards,
    #     W_regularizer=l1l2(l1=l1_reg, l2=l2_reg), U_regularizer=l1l2(l1=l1_reg, l2=l2_reg),
    #     b_regularizer=l1l2(l1=l1_reg, l2=l2_reg), dropout_W=dropout_W, dropout_U=dropout_U))
    model.add(Flatten())
    model.add(Dense(1, W_regularizer=l1l2(l1_reg, l2_reg)))
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
