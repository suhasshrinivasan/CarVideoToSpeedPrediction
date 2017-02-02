from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input as vgg16_preprocess_input
from keras.applications.vgg19 import VGG19
from keras.applications.vgg19 import preprocess_input as vgg19_preprocess_input
from keras.applications.resnet50 import ResNet50
from keras.applications.resnet50 import preprocess_input as resnet50_preprocess_input
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import preprocess_input as inceptionv3_preprocess_input
from keras.preprocessing import image
import numpy as np
import os

def extract_features(data_filepath, num_frames, extraction_network='resnet50', override_existing=True):
    """
    Given the original data filepath and the number of frames in that data,
    this function extracts key image features using ImageNet pre-trained network,
    saving the result to a file.
    If an extracted features file for this data already exists, this function does nothing.

    Credit: https://keras.io/applications/
    """
    extraction_network = str.lower(extraction_network)
    features_filepath = data_filepath[:-4] + '_' + extraction_network + '.npz'
    if os.path.exists(features_filepath) and not override_existing:
        print('Frames converted to extracted features already.')
        return

    print('Extracting Features using pre-trained ' + extraction_network + ' network')
    if extraction_network[:3] == 'vgg' and extraction_network[-2:] == '16':  # 7x7x512
        model = VGG16(weights='imagenet', include_top=False)
        preprocess_input = vgg16_preprocess_input
        target_size = (224, 224)
    elif extraction_network[:3] == 'vgg' and extraction_network[-2:] == '19':  # 7x7x512
        model = VGG19(weights='imagenet', include_top=False)
        preprocess_input = vgg19_preprocess_input
        target_size = (224, 224)
    elif extraction_network[:3] == 'inception' and extraction_network[-2:] == 'v3':  # 8x8x2048
        model = InceptionV3(weights='imagenet', include_top=False)
        preprocess_input = inceptionv3_preprocess_input
        target_size = (299, 299)
    else:  # 1x1x2048
        model = ResNet50(weights='imagenet', include_top=False)
        preprocess_input = resnet50_preprocess_input
        target_size = (224, 224)
        extraction_network = 'resnet50'
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
    print('- Running Network... (This takes on the order of 0.4s per frame on CPU)')
    features = model.predict(x)

    if extraction_network == 'resnet50':
        features = features.reshape(features.shape[0], -1)  # Flatten feature vector per frame

    print('- Saving Features...')
    np.savez_compressed(features_filepath, features)

    print('Frames converted to extracted features.')
    return
