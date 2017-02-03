SPEED PREDICTION

Predicting speed of a car, given car-perspective, recorded video.

Final model:
- Converts MP4 File to series of JPEG frames
- Extract Features from frames using pre-trained Residual Network 50
- Performs Ridge Regression to predict speed at each frame
- Smooths resulting speed prediction vector via a moving average method
- Test Set MSE of 6.31 (2.51 RMSE m/s)
- Train Set MSE of 3.84 (1.96 RMSE m/s)

Be sure to check out cool_viz.mp4 for a cool visualization!

Here are the main modules I used to produce proper runs and results:
- ***Keras 1.2.1*** Important: Other versions (particularly 1.1)
    might not extract features identically.
- ***Theano 0.8.2*** Same story as above.
- Python 2.7
- Numpy 1.12.0
- SKLearn 0.18
- CV2 2.4.9
Other versions might work as well but haven't been tested.
Tensorflow 0.12.1 might work as Keras backend as well, but hasn't been rigorously tested.

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
- Should receive message "Test MSE: 4.21" printed to terminal.
- If not, use Keras 1.2.1 and Theano 0.8.2 as Keras backend.
    Again, feature extraction with pre-trained model might be sensitive to these.

To reproduce final run:
- Run "python predict.py"

Credit Due To:
- Joseph Redmon's YOLO Darknet repo for help generating the visualization.
- Keras, SKLearn, Numpy, etc. Documentation
- Lots of Stack Overflow
- Specific functions built off code from elsewhere is cited in comments and docstrings.
