import cv2
import numpy as np

img = cv2.imread('nctu_flag.jpg')

for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        if img[i][j][0] > 65 and img[i][j][0] * 0.8 > img[i][j][1] and img[i][j][0] * 0.8 > img[i][j][2]:
            continue
        else:
            img[i][j][:] = np.sum(img[i][j]) / 3



cv2.imshow('My Image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()