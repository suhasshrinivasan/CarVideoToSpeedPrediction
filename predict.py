import json
from keras.callbacks import EarlyStopping, TensorBoard
from keras.models import Sequential
from keras.layers import Dense, GRU, LSTM, SimpleRNN
from keras.layers.core import Reshape
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l1l2
import numpy as np
import os
import pandas as pd
from sklearn.metrics import mean_squared_error
import random
import time
import warnings

from frames_to_features import extract_features
from video_to_frames import get_video_frames
from model_testing import hp_sweep, init_model

# Initial Setup
using = 'cpu'
data_dir = 'data/'
filename_base = 'drive'
data_filename_base = os.path.join(data_dir, filename_base)
warnings.filterwarnings(action='ignore', module='scipy', message='^internal gelsd')

# Get feature data
data_filepath = data_filename_base + '.mp4'
num_frames = get_video_frames(data_filepath, override_existing=False)
extract_features(data_filepath, num_frames, model_type='resnet50', override_existing=False)
npz_file = np.load(data_filename_base + '.npz')
X = npz_file['arr_0']

# Get labels
with open (data_filename_base + '.json', 'r') as json_raw_data:
    time_speed_data = json_raw_data.readlines()[0]
time_speed_data = np.array(json.loads(time_speed_data))

# Split data
assert(X.shape[0] == time_speed_data.shape[0])
split_fraction = 0.8
num_train = int(X.shape[0] * split_fraction)
X_train = X[:num_train,:]
X_test = X[num_train:,:]
time_speed_data_train = time_speed_data[:num_train,:]
time_speed_data_test = time_speed_data[num_train:,:]
time_train = time_speed_data_train[:,0].reshape(-1,1)
y_train = time_speed_data_train[:,1].reshape(-1,1)
time_test = time_speed_data_test[:,0].reshape(-1,1)
y_test = time_speed_data_test[:,1].reshape(-1,1)
print('Data processed.')

### Predict
# Simple Model
best_config = {'model_type': 'ridge', 'alpha': 3000}  # 8.44 MSE
if best_config is None:
    best_config = hp_sweep('ridge', X_train, y_train, X_test, y_test)
model = init_model(best_config)
model.fit(X_train, y_train)

for smoothing_window_size in [139]:
    print(smoothing_window_size)

    y_train_pred = model.predict(X_train)
    print('Simple Model Train MSE: %.2f' % mean_squared_error(y_train, y_train_pred))

    y_train_pred_smoothed = pd.rolling_mean(
        y_train_pred, smoothing_window_size, min_periods=1, center=True)
    missing_idxs = np.isnan(y_train_pred_smoothed)
    y_train_pred_smoothed[missing_idxs] = y_train_pred[missing_idxs]
    print('Simple Model Smoothed Train MSE: %.2f' % mean_squared_error(y_train, y_train_pred_smoothed))

    y_test_pred = model.predict(X_test)
    print('Simple Model Test MSE: %.2f' % mean_squared_error(y_test, y_test_pred))

    y_test_pred_smoothed = pd.rolling_mean(
        y_test_pred, smoothing_window_size, min_periods=1, center=True)
    missing_idxs = np.isnan(y_test_pred_smoothed)
    y_test_pred_smoothed[missing_idxs] = y_test_pred[missing_idxs]
    print('Simple Model Smoothed Test MSE: %.2f' % mean_squared_error(y_test, y_test_pred_smoothed))


# Set Up Neural Network Model
X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))

l1_reg = 0.0
l2_reg = 0.0
dropout_W = 0.5
dropout_U = 0.5
bn = True
nb_epochs = 100
batch_size = 10
go_backwards = True
validation_split = 0.05
patience = 3

rnn_config = (l1_reg, l2_reg, dropout_W, dropout_U, bn, nb_epochs, batch_size, go_backwards)
rnn_config_str = '/l1_reg=' + str(l1_reg) + '/l2_reg=' + str(l2_reg) + '/dropout_W=' + str(dropout_W)\
    + '/dropout_U=' + str(dropout_U) + '/bn=' + str(bn) + '/nb_epochs=' + str(nb_epochs)\
    + '/batch_size=' + str(batch_size) + '/go_backwards=' + str(go_backwards)\
    + '/validation_split=' + str(validation_split) + '/patience=' + str(patience) + '/'\
    + str(random.randint(1, 1000000))
callbacks = [
    EarlyStopping(monitor='val_loss', patience=patience, verbose=0, mode='auto'),
    TensorBoard(log_dir='./logs' + rnn_config_str, histogram_freq=0,
        write_graph=True, write_images=False),
]
print(rnn_config_str)

# Compile and Train Neural Network Model
model = Sequential()
model.add(GRU(100, unroll=True, consume_less=using,
    input_dim=X_train.shape[-1], input_length=1, go_backwards=go_backwards,
    W_regularizer=l1l2(l1=l1_reg, l2=l2_reg), U_regularizer=l1l2(l1=l1_reg, l2=l2_reg),
    b_regularizer=l1l2(l1=l1_reg, l2=l2_reg), dropout_W=dropout_W, dropout_U=dropout_U))
if bn:
    model.add(BatchNormalization())
# model.add(Dense(10, activation='relu'))
# model.add(BatchNormalization())
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')

t = time.time()
model.fit(X_train, y_train, nb_epoch=nb_epochs, batch_size=batch_size,
    validation_split=validation_split, verbose=2, callbacks=callbacks)
print('Training Time: %.2f' % (time.time()-t))

# Test Neural Network Model
print(rnn_config_str)

for smoothing_window_size in [1, 3, 5, 9, 19, 39, 59, 79, 99, 119, 139, 159, 179, 199, 219, 239, 259, 279, 299]:
    print(smoothing_window_size)

    y_train_pred = model.predict(X_train)
    print('Complex Model Train MSE: %.2f' % mean_squared_error(y_train, y_train_pred))

    y_train_pred_smoothed = pd.rolling_mean(
        y_train_pred, smoothing_window_size, min_periods=1, center=True)
    missing_idxs = np.isnan(y_train_pred_smoothed)
    y_train_pred_smoothed[missing_idxs] = y_train_pred[missing_idxs]
    print('Simple Model Smoothed Train MSE: %.2f' % mean_squared_error(y_train, y_train_pred_smoothed))

    y_test_pred = model.predict(X_test)
    print('Complex Model Test MSE: %.2f' % mean_squared_error(y_test, y_test_pred))

    y_test_pred_smoothed = pd.rolling_mean(
        y_test_pred, smoothing_window_size, min_periods=1, center=True)
    missing_idxs = np.isnan(y_test_pred_smoothed)
    y_test_pred_smoothed[missing_idxs] = y_test_pred[missing_idxs]
    print('Simple Model Smoothed Test MSE: %.2f' % mean_squared_error(y_test, y_test_pred_smoothed))
