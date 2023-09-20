import cv2
import numpy as np

ENLARGE_SIZE = 3
filename = './Lena_52x52.jpg'

img = cv2.imread(filename)

new_img = np.zeros((img.shape[0] * ENLARGE_SIZE, img.shape[1] * ENLARGE_SIZE, 3), dtype=np.uint8)

for i in range(new_img.shape[0]):
    for j in range(new_img.shape[1]):
        y = int(i / ENLARGE_SIZE)
        x = int(j / ENLARGE_SIZE)
        new_img[i, j] = img[y, x]


winname = 'NearestNeighbor'

cv2.namedWindow(winname)        # Create a named window
cv2.moveWindow(winname, 40, 30)  # Move it to (40,30)
cv2.imshow(winname, new_img)
cv2.waitKey(0)
cv2.destroyAllWindows()