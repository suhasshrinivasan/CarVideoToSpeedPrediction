print('echo "Starting to generate cool visualizations!"')
# print('mkdir viz')
for i in range(8010, 8616):
    print('./darknet detector test cfg/coco.data cfg/yolo.cfg yolo.weights drive/frame%d.jpg' % i)
    print('mv predictions.jpg viz/frame%d.jpg' % i)
print('ffmpeg -pix_fmt yuv420p -framerate 24 -i viz/frame%d.jpg -f mp4 -vcodec h264 cool_viz.mp4')
print('echo "Done generating cool visualizations!""')
