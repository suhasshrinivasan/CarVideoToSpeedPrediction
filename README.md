SPEED PREDICTION

Predicting speed of a car, given car-perspective, recorded video.

Final model:
- Converts MP4 File to series of JPEG frames
- Extract Features from frames using pre-trained Residual Network 50
- Performs Ridge Regression to predict speed at each frame
- Smooths resulting speed prediction vector via a moving average method
- Test Set MSE of 4.22 (2.05 RMSE m/s)
- Train Set MSE of 2.97 (1.72 RMSE m/s)

Be sure to check out cool_viz.mp4 for a cool visualization!

Here are the main modules I used to produce proper runs and results:
- ***Keras 1.1.0*** Important: Some other versions (particularly 1.2)
    do not extract features identically. You have been warned :)
- ***Tensorflow 0.10.0*** for Keras backend. Potentially same story as above.
- Python 2.7
- Numpy 1.11.1
- SKLearn 0.18
- CV2 2.4.11
Other versions (particularly for the last four) might work as well but haven't been tested.

To verify running properly:
- Please do this if possible to confirm feature extraction is working
    as expected under your environment
- Run "python test.py -v"
- This will run the model over the combined training and testing data used originally.
- Should receive message "Test MSE: 3.17" printed to terminal.
- If not, use Keras 1.1.0 and Tensorflow 0.10.0 as Keras backend.
    Again, feature extraction with pre-trained model is sensitive to these.

To test on new data:
- Name the video file "drive_test.mp4"
- Name the accompanying speed labels file "drive_test.json"
- From project's top level directory, run "python test.py"
- Download and change versions of required packages as necessary if errors occur
- If execution is interrupted while converting "drive_test.mp4" into the folder of JPEG's
    "drive/", please delete the created "drive/" folder and run "python test.py" again.

To reproduce final run:
- Run "python predict.py"

Credit Due To:
- Joseph Redmon's YOLO Darknet repo for help generating the visualization.
- Keras, SKLearn, Numpy, etc. Documentation
- Lots of Stack Overflow
- Specific functions built off code from elsewhere is cited in comments and docstrings.
