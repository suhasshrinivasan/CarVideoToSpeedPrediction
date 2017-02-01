from keras.applications.resnet50 import ResNet50
from keras.applications.resnet50 import preprocess_input as resnet50_preprocess_input
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import preprocess_input as inceptionv3_preprocess_input
# from keras.applications.xception import Xception
# from keras.applications.xception import preprocess_input as xception_preprocess_input
from keras.preprocessing import image
import numpy as np
import os

from video_to_frames import get_video_frames

def extract_features(data_filepath, num_frames, extraction_network='resnet50', override_existing=True):
    """
    Given the original data filepath and the number of frames in that data,
    this function extracts key image features using ImageNet pre-trained network,
    saving the result to a file.
    If an extracted features file for this data already exists, this function does nothing.

    Credit: https://keras.io/applications/
    """
    extraction_network = str.lower(extraction_network)
    features_filepath = data_filepath[:-4] + '.npz'
    if os.path.exists(features_filepath) and not override_existing:
        print('Frames converted to extracted features already.')
        return

    print('Extracting Features using pre-trained ' + extraction_network + ' network')
    if extraction_network == 'inceptionv3':
        model = InceptionV3(weights='imagenet', include_top=False)
        preprocess_input = inceptionv3_preprocess_input
        target_size = (299, 299)
    elif extraction_network == 'xception':
        model = Xception(weights='imagenet', include_top=False)
        preprocess_input = xception_preprocess_input
        target_size = (299, 299)
    else:
        model = ResNet50(weights='imagenet', include_top=False)
        preprocess_input = resnet50_preprocess_input
        target_size = (224, 224)
    print('Target Size: ' + str(target_size))

    # Convert JPEG's to PIL Image Instances
    imgs = []
    for i in range(num_frames):
        if i % 100 == 0:
            print 'Converting JPEG to PIL for frame' + str(i) + '.jpg...'
        img_path = data_filepath[:-4] + '/frame%d.jpg' % i
        img = image.load_img(img_path, target_size=target_size)
        imgs.append(img)

    print('- Converting PIL Images to Numpy Array')
    x = []
    for i in range(num_frames):
        x.append(image.img_to_array(imgs[i]))
    x = np.array(x)

    print('- Preprocessing Data...')
    x = preprocess_input(x)

    print('- Running Network... (This might take up to several hours)')
    features = model.predict(x)
    if extraction_network == 'resnet50':
        features = features.reshape(features.shape[0], -1)  # Flatten feature vector per frame

    print('- Saving Features...')
    np.savez_compressed(features_filepath, features)

    print('Frames converted to extracted features.')
    return
