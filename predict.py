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
import cPickle as pickle
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import random
import sys
import time
import warnings

from frames_to_features import extract_features
from model_testing import hp_sweep, init_model
from smooth_signal import ma_smoothing, ewma_smoothing
from video_to_frames import get_video_frames

matplotlib.use('qt5agg')
import matplotlib.pyplot as plt

# A Few Important Configurable Variables
model_type = 'ridge'
extraction_network = 'resnet50'
smooth_signal = ma_smoothing
smooth_data = ma_smoothing
smooth_data_window_size = 1
smooth_signal_window_sizes = 153  # Set to a list of values to test them via validation OR to a value to set it constant
scale_data = True
show_model_plots = True
best_config = {'model_type': 'ridge', 'alpha': 20000.0}  # alpha=3000 good too. None for HP Sweep or non-final runs
best_pca_n_components = 0  # <= num features, 0 for No PCA (best for ridge), None to HP Sweep
k_fold = 10  # For Cross-Validation. Needs to be >5 for training folds to have enough data
val_fraction = 0.15  # What fraction of the training data to use for validation
train_fraction = 0.839  # Split for train+val chosen to equalize ratio of highway to city driving in training and testing data
val_data_location = None  # What section ('beg', 'end', or None) the validation comes from within the training data. None to train and test final model.
test_data_location = 'end'  # What section ('beg' or 'end') test data comes from withing full data
overwrite_final_model = False

# Initial Setup
using = 'cpu'
if len(sys.argv) >= 2 and (sys.argv[-1] == '-g' or sys.argv[1] == '--gpu-optimized'):
    using = 'gpu'
model_type = str.lower(model_type)
data_dir = 'data/'
filename_base = 'drive_orig_theano'
data_filename_base = os.path.join(data_dir, filename_base)
features_filepath = data_filename_base + '_' + extraction_network + '.npz'
warnings.filterwarnings(action='ignore', module='scipy', message='^internal gelsd')
print ('Set to train ' + str.upper(model_type) + ' Model.')

# Get feature data
data_filepath = data_filename_base + '.mp4'
num_frames = get_video_frames(data_filepath, override_existing=False)
extract_features(
    data_filepath, num_frames, extraction_network=extraction_network, override_existing=False)
npz_file = np.load(features_filepath)
X = npz_file['arr_0']
print('Extracted Feature Data Shape: ' + str(X.shape))

# Get labels
with open (data_filename_base + '.json', 'r') as json_raw_data:
    time_speed_data = json_raw_data.readlines()[0]
time_speed_data = np.array(json.loads(time_speed_data))

# Split data
time_speed_data = time_speed_data[:X.shape[0],:]
num_train = int(X.shape[0] * train_fraction)
num_test = X.shape[0] - num_train

if test_data_location == 'beg':
    X_train = X[num_test:,:]
    X_test = X[:num_test,:]
    time_speed_data_train = time_speed_data[num_test:,:]
    time_speed_data_test = time_speed_data[:num_test,:]
else:  # Select from 'end' by default
    X_train = X[:num_train,:]
    X_test = X[num_train:,:]
    time_speed_data_train = time_speed_data[:num_train,:]
    time_speed_data_test = time_speed_data[num_train:,:]

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
            pca_n_components_list = [64, 128, 256, 512, 1024, 2048]
        else:
            pca_n_components_list = [best_pca_n_components]
        best_config, best_train_val_error = hp_sweep(
            model_type, X_train, y_train, k_fold, scale_data, pca_n_components_list)

    # Use Validation Data to Select Smoothing
    if val_data_location == 'beg':
        num_val = int(X_train.shape[0] * val_fraction)
        X_test = X_train[:num_val,:].copy()
        X_train = X_train[num_val:,:]
        y_test = y_train[:num_val,:].copy()
        y_train = y_train[num_val:,:]
        print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)
    elif val_data_location == 'end':
        num_val = int(X_train.shape[0] * val_fraction)
        num_train = X_train.shape[0] - num_val
        print num_val, num_train
        X_test = X_train[num_train:,:].copy()
        X_train = X_train[:num_train,:]
        y_test = y_train[num_train:,:].copy()
        y_train = y_train[:num_train,:]
        print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

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

    if not isinstance(smooth_signal_window_sizes, (list, tuple)):
        smooth_signal_window_sizes = [smooth_signal_window_sizes]

    for smooth_signal_window_size in smooth_signal_window_sizes:
        print smooth_signal_window_size
        y_train_pred = model.predict(X_train)
        print(str.upper(model_type) +
            ' Model Train MSE: %.2f' % mean_squared_error(y_train, y_train_pred))

        y_train_pred_smoothed = smooth_signal(y_train_pred, smooth_signal_window_size)
        print(str.upper(model_type) +
            ' Model Smoothed Train MSE: %.2f' % mean_squared_error(y_train, y_train_pred_smoothed))

        y_test_pred = model.predict(X_test)
        print(str.upper(model_type) +
            ' Model Test MSE: %.2f' % mean_squared_error(y_test, y_test_pred))

        y_test_pred_smoothed = smooth_signal(y_test_pred, smooth_signal_window_size)
        print(str.upper(model_type) +
            ' Model Smoothed Test MSE: %.2f' % mean_squared_error(y_test, y_test_pred_smoothed))

        if show_model_plots:
            plt.plot(y_train,label='Actual')
            plt.plot(y_train_pred, label='Predicted')
            plt.plot(y_train_pred_smoothed, label='Predicted Smoothed')
            plt.title('Training Data: 3.84 MSE')
            plt.savefig('train_error_plot.png')
            plt.show()

            plt.plot(y_test,label='Actual')
            plt.plot(y_test_pred, label='Predicted')
            plt.plot(y_test_pred_smoothed, label='Predicted Smoothed')
            plt.title('Testing Data: 6.31 MSE')
            plt.savefig('test_error_plot.png')
            plt.show()

    if overwrite_final_model:
    	with open('final_model.pickle', 'wb') as f:
    		pickle.dump(model, f, -1)

    	with open('final_scaler.pickle', 'wb') as f:
    		pickle.dump(scaler, f, -1)

else:  # Neural Network Model
    print('Keras model training optimized for ' + str.upper(using) + '.')

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
    model.add(Convolution2D(
        num_hidden_units, 4, 4,
        border_mode='valid',
        subsample=(2, 2),
        input_shape=(8, 8, 2048),
        W_regularizer=l1l2(l1_reg, l2_reg),
    ))
    # model.add(GRU(100, unroll=True, consume_less=using,
    #     input_dim=X_train.shape[-1], input_length=1, go_backwards=go_backwards,
    #     W_regularizer=l1l2(l1=l1_reg, l2=l2_reg), U_regularizer=l1l2(l1=l1_reg, l2=l2_reg),
    #     b_regularizer=l1l2(l1=l1_reg, l2=l2_reg), dropout_W=dropout_W, dropout_U=dropout_U))
    if bn:
        model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(3, 3)))
    if dropout_U > 0:
        model.add(Dropout(dropout_U))
    model.add(Flatten())
    model.add(Dense(1, W_regularizer=l1l2(l1_reg, l2_reg)))
    model.compile(loss='mean_squared_error', optimizer='adam')

    t = time.time()
    hist = model.fit(X_train, y_train, nb_epoch=nb_epochs, batch_size=batch_size,
        validation_split=validation_split, verbose=2, callbacks=callbacks)
    print('Training Time: %.2f' % (time.time()-t))

    # Test Neural Network Model
    print(nn_config_str)

    for smooth_signal_window_size in [1, 49, 99, 149]:
        print(smooth_signal_window_size)

        y_train_pred = model.predict(X_train)
        print(str.upper(model_type) +
            ' Model Train MSE: %.2f' % mean_squared_error(y_train, y_train_pred))

        y_train_pred_smoothed = smooth_signal(y_train_pred, smooth_signal_window_size)
        print(str.upper(model_type) +
            ' Model Smoothed Train MSE: %.2f' % mean_squared_error(y_train, y_train_pred_smoothed))

        y_test_pred = model.predict(X_test)
        print(str.upper(model_type) +
            ' Model Test MSE: %.2f' % mean_squared_error(y_test, y_test_pred))

        y_test_pred_smoothed = smooth_signal(y_test_pred, smooth_signal_window_size)
        print(str.upper(model_type) +
            ' Model Smoothed Test MSE: %.2f' % mean_squared_error(y_test, y_test_pred_smoothed))

        if show_model_plots:
            plt.plot(y_train,label='Actual Train')
            plt.plot(y_train_pred, label='Predicted')
            plt.plot(y_train_pred_smoothed, label='Predicted Smoothed' + str(smooth_signal_window_size))
            plt.show()

            plt.plot(y_test,label='Actual Train')
            plt.plot(y_test_pred, label='Predicted')
            plt.plot(y_test_pred_smoothed, label='Predicted Smoothed' + str(smooth_signal_window_size))
            plt.show()
