import json
from keras.models import Sequential
from keras.layers import Dense, GRU, LSTM, SimpleRNN
from keras.regularizers import l1l2
import numpy as np
import os
from sklearn.metrics import mean_squared_error
import time
import warnings

from frames_to_features import extract_features
from video_to_frames import get_video_frames
from model_testing import hp_sweep, init_model

# Initial Setup
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
y_test_pred = model.predict(X_test)
print('Simple Model MSE: %.2f' % mean_squared_error(y_test, y_test_pred))

# Complex Model
X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))

go_backwards = True  # Whether or not to predict using data from both directions
model = Sequential()
model.add(GRU(200, return_sequences=False, unroll=True, consume_less='cpu',
    input_dim=X_train.shape[-1], input_length=1, go_backwards=go_backwards,
    W_regularizer=l1l2(l1=0.01, l2=0.01), U_regularizer=l1l2(l1=0.01, l2=0.01),
    b_regularizer=l1l2(l1=0.01, l2=0.01), dropout_W=0.5, dropout_U=0.5))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
t = time.time()
model.fit(X_train, y_train, nb_epoch=50, batch_size=300, validation_split=0.1,
    verbose=2)
print(time.time()-t)

y_train_pred = model.predict(X_train)
print('Complex Model Train MSE: %.2f' % mean_squared_error(y_train, y_train_pred))

y_test_pred = model.predict(X_test)
print('Complex Model Test MSE: %.2f' % mean_squared_error(y_test, y_test_pred))
