import cv2
import os

# Add a filename w/o extension here so it will create a folder with the same name
filename = 'tushar'
os.mkdir(filename)
print(cv2.__version__)  # my version is 3.1.0
vidcap = cv2.VideoCapture(filename + '.mp4')
count = 0
while True:
    success,image = vidcap.read()
    if not success:
        break
    cv2.imwrite(os.path.join(filename,"frame{:d}.jpg".format(count)), image)     # save frame as JPEG file
    count += 1
print("{} images are extacted in {}.".format(count,filename))
