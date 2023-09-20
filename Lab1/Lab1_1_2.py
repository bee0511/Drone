import cv2
import numpy as np
import os

img = cv2.imread('nctu_flag.jpg')

CONTRAST = 100
BRIGHTNESS = 40

img = np.array(img, dtype=np.int32)

for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        b, g, r = img[i, j]
        if ((b + g) * 0.3 > r):
            img[i, j] = np.clip((img[i, j] - 127) * (CONTRAST / 127 + 1) + 127 + BRIGHTNESS, 0, 255)
        
img = np.uint8(img)

cv2.imshow('My Image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()