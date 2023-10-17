import numpy as np
import cv2 

img = cv2.imread('test.jpg')
rect = cv2.selectROI("Select desired area.", img)

mask = np.zeros((img.shape[0], img.shape[1]), np.uint8)
bgdModel = np.zeros((1,65), np.float64)
fgdModel = np.zeros((1,65), np.float64)
iter = 15 # iterations

cv2.grabCut(img, mask, rect, bgdModel, fgdModel, iter, cv2.GC_INIT_WITH_RECT)
grabcut_mask = np.where((mask==2)|(mask==0), 0, 1).astype('uint8')
# add a new dimension to grabcut_mask (2D -> 3D) because img is 3D
img = img * grabcut_mask[:, :, np.newaxis] 

cv2.imshow("Image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()