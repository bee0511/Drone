import cv2
import numpy as np
import os

ENLARGE_SIZE = 10

filename = './Lena_52x52.jpg'

img = cv2.imread(filename)

new_img = np.zeros((img.shape[0] * ENLARGE_SIZE, img.shape[1] * ENLARGE_SIZE, 3), dtype=np.uint8)

m = new_img.shape[0] / (img.shape[0] - 1)
n = new_img.shape[1] / (img.shape[1] - 1)
# print (m, n)

for i in range(new_img.shape[0]):
    for j in range(new_img.shape[1]):
        y = int(i / m)
        x = int(j / n)
        a = (i - y * m) / m
        b = (j - x * n) / n
        # print(y, x, a, b)
        # os.system("pause")
        new_img[i, j] = ((1 - a) * (1 - b) * img[y, x]) + a * (1 - b) * img[y + 1, x] + b * (1 - a) * img[y, x + 1] + a * b * img[y + 1, x + 1]

winname = "Bilinear"

cv2.namedWindow(winname)         # Create a named window
cv2.moveWindow(winname, 40, 30)  # Move it to (40,30)
cv2.imshow(winname, new_img)
cv2.waitKey(0)
cv2.destroyAllWindows()