import cv2
import os

def get_video_frames(data_filepath, override_existing=True):
    """
    Takes in a video MP4 file, given its filepath, and converts it to
    a series of ordered JPEG files, collected in a folder with a corresponding name.
    Returns the number of frames read in.
    If a frames directory already exists, just returns the number of files in that directory.

    Credit: http://stackoverflow.com/questions/33311153/python-extracting-and-saving-video-frames
    """
    data_frames_dir = data_filepath[:-4]
    if os.path.exists(data_frames_dir) and not override_existing:
        print('Data converted to frames already.')
    else:
        os.makedirs(data_frames_dir)
        vidcap = cv2.VideoCapture(data_filepath)
        success,image = vidcap.read()
        count = 0
        success = True
        while success:
            success, image = vidcap.read()
            cv2.imwrite(data_frames_dir + '/frame%d.jpg' % count, image)  # Save frame as JPEG
            if count % 100 == 0:
                print('Read frame%d.jpg' % count)
            if cv2.waitKey(10) == 27:  # Exit if Escape is hit
                break
            count += 1
        print('Data converted to frames.')

    return len(filter(
        lambda frame: frame[:5] == 'frame' and frame[-4:] == '.jpg', os.listdir(data_frames_dir)))
