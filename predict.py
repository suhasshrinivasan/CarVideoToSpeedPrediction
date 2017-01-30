import json
from keras.callbacks import EarlyStopping, TensorBoard
from keras.models import Sequential
from keras.layers import Dense, GRU, LSTM, SimpleRNN
from keras.layers.core import Reshape
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l1l2
import matplotlib
import numpy as np
import os
from sklearn.metrics import mean_squared_error
import random
import sys
import time
import warnings

from frames_to_features import extract_features
from video_to_frames import get_video_frames
from model_testing import hp_sweep, init_model
from smooth_signal import ma_smoothing, ewma_smoothing

# Initial Setup
matplotlib.use('qt5agg')
import matplotlib.pyplot as plt
smooth_signal = ma_smoothing
using = 'cpu'
if len(sys.argv) == 2:
    if sys.argv[1] == '-g' or sys.argv[1] == '--gpu-optimized':
        using = 'gpu'
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
split_fraction = 0.9
num_train = int(X.shape[0] * split_fraction)
num_test = X.shape[0] - num_train
X_train = X[:num_train,:]
X_test = X[num_train:,:]
# X_train = X[num_test:,:]
# X_test = X[:num_test,:]
time_speed_data_train = time_speed_data[:num_train,:]
time_speed_data_test = time_speed_data[num_train:,:]
# time_speed_data_train = time_speed_data[num_test:,:]
# time_speed_data_test = time_speed_data[:num_test,:]
time_train = time_speed_data_train[:,0].reshape(-1,1)
y_train = time_speed_data_train[:,1].reshape(-1,1)
time_test = time_speed_data_test[:,0].reshape(-1,1)
y_test = time_speed_data_test[:,1].reshape(-1,1)

# Additional preprocessing: row-wise differences
data_smoothing_window_size = 135
X_train = ma_smoothing(X_train, data_smoothing_window_size)
X_test = ma_smoothing(X_test, data_smoothing_window_size)
print('Data processed.')

### Predict
# Simple Model
# best_config = {'model_type': 'ridge', 'alpha': 3000}  # 8.44 MSE
best_config = None
if best_config is None:
    best_config, best_train_val_error = hp_sweep('ridge', X_train, y_train, X_test, y_test)
model = init_model(best_config)
model.fit(X_train, y_train)

for smoothing_window_size in [1]:
    print(smoothing_window_size)

    y_train_pred = model.predict(X_train)
    print('Simple Model Train MSE: %.2f' % mean_squared_error(y_train, y_train_pred))

    y_train_pred_smoothed = smooth_signal(y_train_pred, smoothing_window_size)
    print('Simple Model Smoothed Train MSE: %.2f' % mean_squared_error(y_train, y_train_pred_smoothed))
    plt.plot(y_train,label='Actual')
    plt.plot(y_train_pred, label='Predicted')
    plt.plot(y_train_pred_smoothed, label='Predicted Smoothed')
    plt.show()

    y_test_pred = model.predict(X_test)
    print('Simple Model Test MSE: %.2f' % mean_squared_error(y_test, y_test_pred))

    y_test_pred_smoothed = smooth_signal(y_test_pred, smoothing_window_size)
    print('Simple Model Smoothed Test MSE: %.2f' % mean_squared_error(y_test, y_test_pred_smoothed))
    plt.plot(y_test,label='Actual')
    plt.plot(y_test_pred, label='Predicted')
    plt.plot(y_test_pred_smoothed, label='Predicted Smoothed')
    plt.show()

# Set Up Neural Network Model
print('Keras model training optimized for ' + using + '.')
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
patience = 10

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
hist = model.fit(X_train, y_train, nb_epoch=nb_epochs, batch_size=batch_size,
    validation_split=validation_split, verbose=2, callbacks=callbacks)
print(hist.history)
print('Training Time: %.2f' % (time.time()-t))

# Test Neural Network Model
print(rnn_config_str)

for smoothing_window_size in [19, 59, 99, 139, 159, 179, 199, 219, 259]:
    print(smoothing_window_size)

    y_train_pred = model.predict(X_train)
    print('Complex Model Train MSE: %.2f' % mean_squared_error(y_train, y_train_pred))

    y_train_pred_smoothed = smooth_signal(y_train_pred, smoothing_window_size)
    print('Complex Model Smoothed Train MSE: %.2f' % mean_squared_error(y_train, y_train_pred_smoothed))

    y_test_pred = model.predict(X_test)
    print('Complex Model Test MSE: %.2f' % mean_squared_error(y_test, y_test_pred))

    y_test_pred_smoothed = smooth_signal(y_test_pred, smoothing_window_size)
    print('Complex Model Smoothed Test MSE: %.2f' % mean_squared_error(y_test, y_test_pred_smoothed))

    plt.plot(y_train,label='Actual Train')
    plt.plot(y_train_pred, label='Predicted')
    plt.plot(y_train_pred_smoothed, label='Predicted Smoothed' + str(smoothing_window_size))
    plt.show()

    plt.plot(y_test,label='Actual Train')
    plt.plot(y_test_pred, label='Predicted')
    plt.plot(y_test_pred_smoothed, label='Predicted Smoothed' + str(smoothing_window_size))
    plt.show()
