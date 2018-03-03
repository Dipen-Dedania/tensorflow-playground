import cv2

# Check version
# print(cv2.__version__)

import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('images/elon_musk.jpg',cv2.IMREAD_GRAYSCALE)
cv2.imshow('image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()
