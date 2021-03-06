import cPickle as pickle
import json
import matplotlib
import numpy as np
import os
import sys
import warnings
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

from frames_to_features import extract_features
from smooth_signal import ma_smoothing
from video_to_frames import get_video_frames

matplotlib.use('qt5agg')
import matplotlib.pyplot as plt

# Initial Setup
filename_base = 'drive_test'
if len(sys.argv) >= 2 and sys.argv[-1] == '-v':
    filename_base = 'drive'
extraction_network = 'resnet50'
smooth_signal = ma_smoothing
smooth_signal_window_size = 151
show_model_plots = True
data_dir = 'data/'
data_filename_base = os.path.join(data_dir, filename_base)
features_filepath = data_filename_base + '_' + extraction_network + '.npz'
warnings.filterwarnings(action='ignore', module='scipy', message='^internal gelsd')
print('Testing Model!')

# Get labels
data_filepath = data_filename_base + '.mp4'
with open (data_filename_base + '.json', 'r') as json_raw_data:
    time_speed_data = json_raw_data.readlines()[0]
time_speed_data = np.array(json.loads(time_speed_data))
print('Labels Loaded.')

# Get feature data
num_frames = get_video_frames(data_filepath, override_existing=False)
extract_features(
    data_filepath, num_frames, extraction_network=extraction_network, override_existing=False)
npz_file = np.load(features_filepath)
X_test = npz_file['arr_0']
print('Features Extracted.')

# Split data
time_speed_data = time_speed_data[:X_test.shape[0],:]
time_test = time_speed_data[:,0].reshape(-1,1)
y_test = time_speed_data[:,1].reshape(-1,1)
print('Data processed.')

# Load Final Model and Preprocessor
with open('final_model.pickle', 'rb') as f:
	model = pickle.load(f)
with open('final_scaler.pickle', 'rb') as f:
	scaler = pickle.load(f)
print('Model and Preprocessor Loaded.')

# Predict off Extracted Features
X_test = scaler.transform(X_test)
y_test_pred = model.predict(X_test)
y_test_pred_smoothed = smooth_signal(y_test_pred, smooth_signal_window_size)
print('Test MSE: %.2f' % mean_squared_error(y_test, y_test_pred_smoothed))

if show_model_plots:
    plt.plot(y_test,label='Actual')
    plt.plot(y_test_pred, label='Predicted')
    plt.plot(y_test_pred_smoothed, label='Predicted Smoothed')
    plt.show()
