print('echo "Starting to generate cool visualizations!"')
print('mkdir visualization')
for i in range(8616):
    print('./darknet detector test cfg/coco.data cfg/yolo.cfg yolo.weights drive/frame%d.jpg' % i)
    print('mv predictions.jpg visualization/frame%d.jpg' % i)
print('ffmpeg -i frame%d.png -f mp4 -vcodec libx264 -pix_fmt yuv420p test.mp4')
print('echo "Done generating cool visualizations!""')
