SPEED PREDICTION

Predicting speed of a car, given car-perspective, recorded video.

Final model:
- Converts MP4 File to series of JPEG frames
- Extract Features from frames using pre-trained Residual Network 50
- Performs Ridge Regression to predict speed at each frame
- Smooths resulting speed prediction vector via a moving average method
- Test Set MSE of 6.31
- Train Set MSE of 3.84

Here are the main modules I used to produce proper runs and results:
- ***Keras 1.2.1*** Important: Other versions might not extract features identically
- Keras backend: Theano 0.8.2 or Tensorflow 0.12.1
- Python 2.7
- Numpy 1.12.0 or 1.11.1
- SKLearn 0.18
- CV2 2.4.9
Other versions might work as well but haven't been tested.

Be sure to check out cool_viz.mp4 for a cool visualization. Big thanks to Joseph Redmon's YOLO Darknet repo for help generating the visualization.

To test on new data:
- Name the video file "drive_test.mp4"
- Name the accompanying speed labels file "drive_test.json"
- From project's top level directory, run "python test.py"
- Download and change versions of required packages as necessary if errors occur
- If execution is interrupted while converting "drive_test.mp4" into the folder of JPEG's
    "drive/", please delete the created "drive/" folder and run "python test.py" again.

To verify running properly:
- Run "python test.py -v"
- This will run the model over the training and testing data used originally.
- Should receive message "Test MSE: 4.60" printed to terminal.
- If not, use Keras 1.2.1. Feature extraction with pre-trained model might be sensitive to
    Keras version. Also try Theano 0.8.2 or Tensorflow 0.12.1 as Keras backend,
    if not using already.

To reproduce final run:
- Run "python predict.py"
